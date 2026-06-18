import copy
import importlib
import os
import random
from logging import WARNING
from typing import Any, List, Optional, Union

import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
from enum import Enum
import yaml
# from easydict import EasyDict as edict
import cv2
from transformers import BertTokenizer, BertTokenizerFast, T5Tokenizer, ChineseCLIPTextModel, CLIPTextModel


# Copied from easydict
class EasyDict(dict):
    """
    Get attributes

    >>> d = EasyDict({'foo':3})
    >>> d['foo']
    3
    >>> d.foo
    3
    >>> d.bar
    Traceback (most recent call last):
    ...
    AttributeError: 'EasyDict' object has no attribute 'bar'

    Works recursively

    >>> d = EasyDict({'foo':3, 'bar':{'x':1, 'y':2}})
    >>> isinstance(d.bar, dict)
    True
    >>> d.bar.x
    1

    Bullet-proof

    >>> EasyDict({})
    {}
    >>> EasyDict(d={})
    {}
    >>> EasyDict(None)
    {}
    >>> d = {'a': 1}
    >>> EasyDict(**d)
    {'a': 1}
    >>> EasyDict((('a', 1), ('b', 2)))
    {'a': 1, 'b': 2}
    
    Set attributes

    >>> d = EasyDict()
    >>> d.foo = 3
    >>> d.foo
    3
    >>> d.bar = {'prop': 'value'}
    >>> d.bar.prop
    'value'
    >>> d
    {'foo': 3, 'bar': {'prop': 'value'}}
    >>> d.bar.prop = 'newer'
    >>> d.bar.prop
    'newer'
    >>> d.lst = [1, 2, 3]
    >>> d.lst
    [1, 2, 3]
    >>> d.tpl = (1, 2, 3)
    >>> d.tpl
    (1, 2, 3)


    Values extraction

    >>> d = EasyDict({'foo':0, 'bar':[{'x':1, 'y':2}, {'x':3, 'y':4}]})
    >>> isinstance(d.bar, list)
    True
    >>> from operator import attrgetter
    >>> list(map(attrgetter('x'), d.bar))
    [1, 3]
    >>> list(map(attrgetter('y'), d.bar))
    [2, 4]
    >>> d = EasyDict()
    >>> list(d.keys())
    []
    >>> d = EasyDict(foo=3, bar=dict(x=1, y=2))
    >>> d.foo
    3
    >>> d.bar.x
    1

    Still like a dict though

    >>> o = EasyDict({'clean':True})
    >>> list(o.items())
    [('clean', True)]

    And like a class

    >>> class Flower(EasyDict):
    ...     power = 1
    ...     mean = {}
    ...     color = {"r": 100, "g": 0, "b": 0}
    ...
    >>> f = Flower()
    >>> f.power
    1
    >>> f.color.r
    100
    >>> f.mean.x = 10
    >>> f.mean.x
    10
    >>> f = Flower({'height': 12})
    >>> f.height
    12
    >>> f['power']
    1
    >>> sorted(f.keys())
    ['color', 'height', 'mean', 'power']

    update and pop items
    >>> d = EasyDict(a=1, b='2')
    >>> e = EasyDict(c=3.0, a=9.0)
    >>> d.update(e)
    >>> d.c
    3.0
    >>> d['c']
    3.0
    >>> d.get('c')
    3.0
    >>> d.update(a=4, b=4)
    >>> d.b
    4
    >>> d.pop('a')
    4
    >>> d.a
    Traceback (most recent call last):
    ...
    AttributeError: 'EasyDict' object has no attribute 'a'
    >>> d.pop('a', 8)
    8
    >>> d.pop('b', 100)
    4
    >>> d
    {'c': 3.0}
    """
    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        else:
            d = dict(d)        
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith('__') and k.endswith('__')) and k not in ('update', 'pop'):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = type(value)(self.__class__(x)
                     if isinstance(x, dict) else x for x in value)
        elif isinstance(value, dict) and not isinstance(value, EasyDict):
            value = EasyDict(value)
        super(EasyDict, self).__setattr__(name, value)
        super(EasyDict, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k in d:
            setattr(self, k, d[k])

    def pop(self, k, *args):
        if hasattr(self, k):
            delattr(self, k)
        return super(EasyDict, self).pop(k, *args)



def try_import(name: str):
    """Try to import a module.

    Args:
        name (str): Specifies what module to import in absolute or relative
            terms (e.g. either pkg.mod or ..mod).
    Returns:
        ModuleType or None: If importing successfully, returns the imported
        module, otherwise returns None.
    """
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


class TokenizerWrapper:
    """Tokenizer wrapper for CLIPTokenizer. Only support CLIPTokenizer
    currently. This wrapper is modified from https://github.com/huggingface/dif
    fusers/blob/e51f19aee82c8dd874b715a09dbc521d88835d68/src/diffusers/loaders.
    py#L358  # noqa.

    Args:
        from_pretrained (Union[str, os.PathLike], optional): The *model id*
            of a pretrained model or a path to a *directory* containing
            model weights and config. Defaults to None.
        from_config (Union[str, os.PathLike], optional): The *model id*
            of a pretrained model or a path to a *directory* containing
            model weights and config. Defaults to None.

        *args, **kwargs: If `from_pretrained` is passed, *args and **kwargs
            will be passed to `from_pretrained` function. Otherwise, *args
            and **kwargs will be used to initialize the model by
            `self._module_cls(*args, **kwargs)`.
    """

    def __init__(self,
                 tokenizer):
        assert isinstance(tokenizer, (BertTokenizer, BertTokenizerFast))
        self.wrapped = tokenizer
        self._from_pretrained = tokenizer.__class__.__name__
        self.token_map = {}

    def __getattr__(self, name: str) -> Any:
        if name == 'wrapped':
            return super().__getattr__('wrapped')

        try:
            return getattr(self.wrapped, name)
        except AttributeError:
            try:
                return super().__getattr__(name)
            except AttributeError:
                raise AttributeError(
                    '\'name\' cannot be found in both '
                    f'\'{self.__class__.__name__}\' and '
                    f'\'{self.__class__.__name__}.tokenizer\'.')

    def try_adding_tokens(self, tokens: Union[str, List[str]], *args,
                          **kwargs):
        """Attempt to add tokens to the tokenizer.

        Args:
            tokens (Union[str, List[str]]): The tokens to be added.
        """
        num_added_tokens = self.wrapped.add_tokens(tokens, *args, **kwargs)
        assert num_added_tokens != 0, (
            f'The tokenizer already contains the token {tokens}. Please pass '
            'a different `placeholder_token` that is not already in the '
            'tokenizer.')

    def get_token_info(self, token: str) -> dict:
        """Get the information of a token, including its start and end index in
        the current tokenizer.

        Args:
            token (str): The token to be queried.

        Returns:
            dict: The information of the token, including its start and end
                index in current tokenizer.
        """
        token_ids = self.__call__(token).input_ids
        start, end = token_ids[1], token_ids[-2] + 1
        return {'name': token, 'start': start, 'end': end}

    def add_placeholder_token(self,
                              placeholder_token: str,
                              *args,
                              num_vec_per_token: int = 1,
                              **kwargs):
        """Add placeholder tokens to the tokenizer.

        Args:
            placeholder_token (str): The placeholder token to be added.
            num_vec_per_token (int, optional): The number of vectors of
                the added placeholder token.
            *args, **kwargs: The arguments for `self.wrapped.add_tokens`.
        """
        output = []
        if num_vec_per_token == 1:
            self.try_adding_tokens(placeholder_token, *args, **kwargs)
            output.append(placeholder_token)
        else:
            output = []
            for i in range(num_vec_per_token):
                ith_token = placeholder_token + f'_{i}'
                self.try_adding_tokens(ith_token, *args, **kwargs)
                output.append(ith_token)

        for token in self.token_map:
            if token in placeholder_token:
                raise ValueError(
                    f'The tokenizer already has placeholder token {token} '
                    f'that can get confused with {placeholder_token} '
                    'keep placeholder tokens independent')
        self.token_map[placeholder_token] = output

    def replace_placeholder_tokens_in_text(self,
                                           text: Union[str, List[str]],
                                           vector_shuffle: bool = False,
                                           prop_tokens_to_load: float = 1.0
                                           ) -> Union[str, List[str]]:
        """Replace the keywords in text with placeholder tokens. This function
        will be called in `self.__call__` and `self.encode`.

        Args:
            text (Union[str, List[str]]): The text to be processed.
            vector_shuffle (bool, optional): Whether to shuffle the vectors.
                Defaults to False.
            prop_tokens_to_load (float, optional): The proportion of tokens to
                be loaded. If 1.0, all tokens will be loaded. Defaults to 1.0.

        Returns:
            Union[str, List[str]]: The processed text.
        """
        if isinstance(text, list):
            output = []
            for i in range(len(text)):
                output.append(
                    self.replace_placeholder_tokens_in_text(
                        text[i], vector_shuffle=vector_shuffle))
            return output

        for placeholder_token in self.token_map:
            if placeholder_token in text:
                tokens = self.token_map[placeholder_token]
                tokens = tokens[:1 + int(len(tokens) * prop_tokens_to_load)]
                if vector_shuffle:
                    tokens = copy.copy(tokens)
                    random.shuffle(tokens)
                text = text.replace(placeholder_token, ' '.join(tokens))
        return text

    def replace_text_with_placeholder_tokens(self, text: Union[str, List[str]]
                                             ) -> Union[str, List[str]]:
        """Replace the placeholder tokens in text with the original keywords.
        This function will be called in `self.decode`.

        Args:
            text (Union[str, List[str]]): The text to be processed.

        Returns:
            Union[str, List[str]]: The processed text.
        """
        if isinstance(text, list):
            output = []
            for i in range(len(text)):
                output.append(
                    self.replace_text_with_placeholder_tokens(text[i]))
            return output

        for placeholder_token, tokens in self.token_map.items():
            merged_tokens = ' '.join(tokens)
            if merged_tokens in text:
                text = text.replace(merged_tokens, placeholder_token)
        return text

    def __call__(self,
                 text: Union[str, List[str]],
                 *args,
                 vector_shuffle: bool = False,
                 prop_tokens_to_load: float = 1.0,
                 **kwargs):
        """The call function of the wrapper.

        Args:
            text (Union[str, List[str]]): The text to be tokenized.
            vector_shuffle (bool, optional): Whether to shuffle the vectors.
                Defaults to False.
            prop_tokens_to_load (float, optional): The proportion of tokens to
                be loaded. If 1.0, all tokens will be loaded. Defaults to 1.0
            *args, **kwargs: The arguments for `self.wrapped.__call__`.
        """
        replaced_text = self.replace_placeholder_tokens_in_text(
            text,
            vector_shuffle=vector_shuffle,
            prop_tokens_to_load=prop_tokens_to_load)

        return self.wrapped.__call__(replaced_text, *args, **kwargs)

    def encode(self, text: Union[str, List[str]], *args, **kwargs):
        """Encode the passed text to token index.

        Args:
            text (Union[str, List[str]]): The text to be encode.
            *args, **kwargs: The arguments for `self.wrapped.__call__`.
        """
        replaced_text = self.replace_placeholder_tokens_in_text(text)
        return self.wrapped(replaced_text, *args, **kwargs)

    def decode(self,
               token_ids,
               return_raw: bool = False,
               *args,
               **kwargs) -> Union[str, List[str]]:
        """Decode the token index to text.

        Args:
            token_ids: The token index to be decoded.
            return_raw: Whether keep the placeholder token in the text.
                Defaults to False.
            *args, **kwargs: The arguments for `self.wrapped.decode`.

        Returns:
            Union[str, List[str]]: The decoded text.
        """
        text = self.wrapped.decode(token_ids, *args, **kwargs)
        if return_raw:
            return text
        replaced_text = self.replace_text_with_placeholder_tokens(text)
        return replaced_text

    def __repr__(self):
        """The representation of the wrapper."""
        s = super().__repr__()
        prefix = f'Wrapped Module Class: {self._module_cls}\n'
        prefix += f'Wrapped Module Name: {self._module_name}\n'
        if self._from_pretrained:
            prefix += f'From Pretrained: {self._from_pretrained}\n'
        s = prefix + s
        return s


class EmbeddingLayerWithFixes(nn.Module):
    """The revised embedding layer to support external embeddings. This design
    of this class is inspired by https://github.com/AUTOMATIC1111/stable-
    diffusion-webui/blob/22bcc7be428c94e9408f589966c2040187245d81/modules/sd_hi
    jack.py#L224  # noqa.

    Args:
        wrapped (nn.Emebdding): The embedding layer to be wrapped.
        external_embeddings (Union[dict, List[dict]], optional): The external
            embeddings added to this layer. Defaults to None.
    """

    def __init__(self,
                 wrapped: nn.Embedding,
                 external_embeddings: Optional[Union[dict,
                                                     List[dict]]] = None):
        super().__init__()
        self.wrapped = wrapped
        self.num_embeddings = wrapped.weight.shape[0]

        self.external_embeddings = []
        if external_embeddings:
            self.add_embeddings(external_embeddings)

        self.trainable_embeddings = nn.ParameterDict()

    @property
    def weight(self):
        """Get the weight of wrapped embedding layer."""
        return self.wrapped.weight

    def check_duplicate_names(self, embeddings: List[dict]):
        """Check whether duplicate names exist in list of 'external
        embeddings'.

        Args:
            embeddings (List[dict]): A list of embedding to be check.
        """
        names = [emb['name'] for emb in embeddings]
        assert len(names) == len(set(names)), (
            'Found duplicated names in \'external_embeddings\'. Name list: '
            f'\'{names}\'')

    def check_ids_overlap(self, embeddings):
        """Check whether overlap exist in token ids of 'external_embeddings'.

        Args:
            embeddings (List[dict]): A list of embedding to be check.
        """
        ids_range = [[emb['start'], emb['end'], emb['name']]
                     for emb in embeddings]
        ids_range.sort()  # sort by 'start'
        # check if 'end' has overlapping
        for idx in range(len(ids_range) - 1):
            name1, name2 = ids_range[idx][-1], ids_range[idx + 1][-1]
            assert ids_range[idx][1] <= ids_range[idx + 1][0], (
                f'Found ids overlapping between embeddings \'{name1}\' '
                f'and \'{name2}\'.')

    def add_embeddings(self, embeddings: Optional[Union[dict, List[dict]]]):
        """Add external embeddings to this layer.

        Use case:

        >>> 1. Add token to tokenizer and get the token id.
        >>> tokenizer = TokenizerWrapper('openai/clip-vit-base-patch32')
        >>> # 'how much' in kiswahili
        >>> tokenizer.add_placeholder_tokens('ngapi', num_vec_per_token=4)
        >>>
        >>> 2. Add external embeddings to the model.
        >>> new_embedding = {
        >>>     'name': 'ngapi',  # 'how much' in kiswahili
        >>>     'embedding': torch.ones(1, 15) * 4,
        >>>     'start': tokenizer.get_token_info('kwaheri')['start'],
        >>>     'end': tokenizer.get_token_info('kwaheri')['end'],
        >>>     'trainable': False  # if True, will registry as a parameter
        >>> }
        >>> embedding_layer = nn.Embedding(10, 15)
        >>> embedding_layer_wrapper = EmbeddingLayerWithFixes(embedding_layer)
        >>> embedding_layer_wrapper.add_embeddings(new_embedding)
        >>>
        >>> 3. Forward tokenizer and embedding layer!
        >>> input_text = ['hello, ngapi!', 'hello my friend, ngapi?']
        >>> input_ids = tokenizer(
        >>>     input_text, padding='max_length', truncation=True,
        >>>     return_tensors='pt')['input_ids']
        >>> out_feat = embedding_layer_wrapper(input_ids)
        >>>
        >>> 4. Let's validate the result!
        >>> assert (out_feat[0, 3: 7] == 2.3).all()
        >>> assert (out_feat[2, 5: 9] == 2.3).all()

        Args:
            embeddings (Union[dict, list[dict]]): The external embeddings to
                be added. Each dict must contain the following 4 fields: 'name'
                (the name of this embedding), 'embedding' (the embedding
                tensor), 'start' (the start token id of this embedding), 'end'
                (the end token id of this embedding). For example:
                `{name: NAME, start: START, end: END, embedding: torch.Tensor}`
        """
        if isinstance(embeddings, dict):
            embeddings = [embeddings]

        self.external_embeddings += embeddings
        self.check_duplicate_names(self.external_embeddings)
        self.check_ids_overlap(self.external_embeddings)

        # set for trainable
        added_trainable_emb_info = []
        for embedding in embeddings:
            trainable = embedding.get('trainable', False)
            if trainable:
                name = embedding['name']
                embedding['embedding'] = torch.nn.Parameter(
                    embedding['embedding'])
                self.trainable_embeddings[name] = embedding['embedding']
                added_trainable_emb_info.append(name)

        added_emb_info = [emb['name'] for emb in embeddings]
        added_emb_info = ', '.join(added_emb_info)
#         print_log(f'Successfully add external embeddings: {added_emb_info}.',
#                   'current')

        if added_trainable_emb_info:
            added_trainable_emb_info = ', '.join(added_trainable_emb_info)
#             print_log(
#                 'Successfully add trainable external embeddings: '
#                 f'{added_trainable_emb_info}', 'current')

    def replace_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Replace external input ids to 0.

        Args:
            input_ids (torch.Tensor): The input ids to be replaced.

        Returns:
            torch.Tensor: The replaced input ids.
        """
        input_ids_fwd = input_ids.clone()
        input_ids_fwd[input_ids_fwd >= self.num_embeddings] = 0
        return input_ids_fwd

    def replace_embeddings(self, input_ids: torch.Tensor,
                           embedding: torch.Tensor,
                           external_embedding: dict) -> torch.Tensor:
        """Replace external embedding to the embedding layer. Noted that, in
        this function we use `torch.cat` to avoid inplace modification.

        Args:
            input_ids (torch.Tensor): The original token ids. Shape like
                [LENGTH, ].
            embedding (torch.Tensor): The embedding of token ids after
                `replace_input_ids` function.
            external_embedding (dict): The external embedding to be replaced.

        Returns:
            torch.Tensor: The replaced embedding.
        """
        new_embedding = []

        name = external_embedding['name']
        start = external_embedding['start']
        end = external_embedding['end']
        target_ids_to_replace = [i for i in range(start, end)]
        ext_emb = external_embedding['embedding']

        # do not need to replace
        if not (input_ids == start).any():
            return embedding

        # start replace
        s_idx, e_idx = 0, 0
        while e_idx < len(input_ids):
            if input_ids[e_idx] == start:
                if e_idx != 0:
                    # add embedding do not need to replace
                    new_embedding.append(embedding[s_idx:e_idx])

                # check if the next embedding need to replace is valid
                actually_ids_to_replace = [
                    int(i) for i in input_ids[e_idx:e_idx + end - start]
                ]
                assert actually_ids_to_replace == target_ids_to_replace, (
                    f'Invalid \'input_ids\' in position: {s_idx} to {e_idx}. '
                    f'Expect \'{target_ids_to_replace}\' for embedding '
                    f'\'{name}\' but found \'{actually_ids_to_replace}\'.')

                new_embedding.append(ext_emb)

                s_idx = e_idx + end - start
                e_idx = s_idx + 1
            else:
                e_idx += 1

        if e_idx == len(input_ids):
            new_embedding.append(embedding[s_idx:e_idx])

        return torch.cat(new_embedding, dim=0)

    def forward(self,
                input_ids: torch.Tensor, 
                external_embeddings: Optional[List[dict]] = None):
        """The forward function.

        Args:
            input_ids (torch.Tensor): The token ids shape like [bz, LENGTH] or
                [LENGTH, ].
            external_embeddings (Optional[List[dict]]): The external
                embeddings. If not passed, only `self.external_embeddings`
                will be used.  Defaults to None.

        input_ids: shape like [bz, LENGTH] or [LENGTH].
        """
        assert input_ids.ndim in [1, 2]
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        if external_embeddings is None and not self.external_embeddings:
            return self.wrapped(input_ids)

        input_ids_fwd = self.replace_input_ids(input_ids) 
        inputs_embeds = self.wrapped(input_ids_fwd)
        vecs = []

        if external_embeddings is None:
            external_embeddings = []
        elif isinstance(external_embeddings, dict):
            external_embeddings = [external_embeddings]
        embeddings = self.external_embeddings + external_embeddings

        for input_id, embedding in zip(input_ids, inputs_embeds): # batch dim
            new_embedding = embedding 
            for external_embedding in embeddings:
                new_embedding = self.replace_embeddings(
                    input_id, new_embedding, external_embedding)
            vecs.append(new_embedding)

        return torch.stack(vecs)


def add_tokens(tokenizer,
               text_encoder,
               placeholder_tokens: list,
               initialize_tokens: list = None,
               num_vectors_per_token: int = 1):
    """Add token for training.

    # TODO: support add tokens as dict, then we can load pretrained tokens.
    """
    if initialize_tokens is not None:
        assert len(initialize_tokens) == len(placeholder_tokens), (
            'placeholder_token should be the same length as initialize_token')
    for ii in range(len(placeholder_tokens)):

        tokenizer.add_placeholder_token(
            placeholder_tokens[ii], num_vec_per_token=num_vectors_per_token)

    # text_encoder.set_embedding_layer()
    assert isinstance(text_encoder, (CLIPTextModel, ChineseCLIPTextModel,))
    if isinstance(text_encoder, CLIPTextModel):
        embedding_layer = text_encoder.text_model.embeddings.token_embedding
        text_encoder.text_model.embeddings.token_embedding = \
            EmbeddingLayerWithFixes(embedding_layer) 
        embedding_layer = text_encoder.text_model.embeddings.token_embedding
    elif isinstance(text_encoder, ChineseCLIPTextModel):
        embedding_layer = text_encoder.embeddings.word_embeddings
        text_encoder.embeddings.word_embeddings = \
            EmbeddingLayerWithFixes(embedding_layer)
        embedding_layer = text_encoder.embeddings.word_embeddings

    assert embedding_layer is not None, (
        'Do not support get embedding layer for current text encoder. '
        'Please check your configuration.')
    initialize_embedding = []
    if initialize_tokens is not None:
        for ii in range(len(placeholder_tokens)):
            init_id = tokenizer(initialize_tokens[ii]).input_ids[1]
            temp_embedding = embedding_layer.weight[init_id]
            initialize_embedding.append(temp_embedding[None, ...].repeat(
                num_vectors_per_token, 1))
    else:
        for ii in range(len(placeholder_tokens)):
            init_id = tokenizer('a').input_ids[1]
            temp_embedding = embedding_layer.weight[init_id]
            len_emb = temp_embedding.shape[0]
            init_weight = (torch.rand(num_vectors_per_token, len_emb) -
                           0.5) / 2.0
            initialize_embedding.append(init_weight)

    token_info_all = []
    for ii in range(len(placeholder_tokens)):
        token_info = tokenizer.get_token_info(placeholder_tokens[ii])
        token_info['embedding'] = initialize_embedding[ii]
        token_info['trainable'] = True
        token_info_all.append(token_info)
    embedding_layer.add_embeddings(token_info_all)


class RandomMaskCrop(torch.nn.Module):
    '''
    random crop mask (must include mask)
    random crop square from mask and image
    Attention: 
        1. must use transforms.Resize to resize image to the same short edge first (short edge == crop size)
        2. mask channel must == 1
    '''
    def __init__(self, size):
        super().__init__()
        self.size = size

    def propose_random_square_crop(self, mask, min_overlap=0.5):
        height, width = mask.shape

        mask_ys, mask_xs = torch.where(mask > 0.5)  # mask==0 is known fragment and mask==1 is missing
        # mask values are all less than 0.5
        if not len(mask_ys):
            if height < width:
                crop_size = height
                start_x = np.random.randint(0, width - crop_size)
                return 0, start_x, height, crop_size
            else:
                crop_size = width
                start_y = np.random.randint(0, height - crop_size) if height > width else 0
                return start_y, 0, crop_size, width
            
        if height < width:
            crop_size = height
            obj_left, obj_right = mask_xs.min(), mask_xs.max()
            obj_width = obj_right - obj_left
            left_border = max(0, min(width - crop_size - 1, obj_left + obj_width * min_overlap - crop_size))
            right_border = max(left_border + 1, min(width - crop_size, obj_left + obj_width * min_overlap))
            start_x = np.random.randint(left_border, right_border)
            return 0, start_x, height, crop_size
        else:
            crop_size = width
            obj_top, obj_bottom = mask_ys.min(), mask_ys.max()
            obj_height = obj_bottom - obj_top
            top_border = max(0, min(height - crop_size - 1, obj_top + obj_height * min_overlap - crop_size))
            bottom_border = max(top_border + 1, min(height - crop_size, obj_top + obj_height * min_overlap))
            start_y = np.random.randint(top_border, bottom_border)
            return start_y, 0, crop_size, width

    def forward(self, imageandmask, min_overlap=0.5):
        mask = imageandmask[-1]
        (crop_top, crop_left, crop_height, crop_width) = self.propose_random_square_crop(mask, min_overlap)
        return TF.crop(imageandmask, crop_top, crop_left, crop_height, crop_width)
    

class BlurMaskShape(torch.nn.Module):
    '''
    control the mask shape by ms
    '''
    def __init__(self, max_kernel_size=50):
        super().__init__()
        self.max_kernel_size = max_kernel_size
        
    def forward(self, mask):
        # input mask shape: (h, w)
        # masked the whole image
        if torch.all(mask > 0.5):
            return mask[None]

        kernel_size = random.randint(0, self.max_kernel_size)
        if kernel_size < 20:
            mask_ori = (mask > 0).to(torch.uint8)
            y_coords, x_coords = torch.nonzero(mask_ori, as_tuple=True)
            if (not len(y_coords) or not len(x_coords)):
                x_min = 0
                x_max = mask_ori.shape[1] - 1
                y_min = 0
                y_max = mask_ori.shape[0] - 1
            else:
                x_min = x_coords.min()
                x_max = x_coords.max()
                y_min = y_coords.min()
                y_max = y_coords.max()
            mask_fill = torch.ones((y_max-y_min, x_max-x_min), dtype=torch.uint8)
            mask_ori[y_min:y_max, x_min:x_max] = mask_fill
            mask = mask_ori[None]
        else:
            kernel_size = kernel_size if kernel_size % 2 !=0 else kernel_size+1
            mask = TF.gaussian_blur(mask[None], (kernel_size, kernel_size))
            # mask = torch.where(mask > 0, 1, 0)
            
        return mask


# generate random masks
def random_mask(im_shape, ratio=1, mask_full_image=False):
    mask = Image.new("L", im_shape, 0)
    draw = ImageDraw.Draw(mask)
    size = (random.randint(0, int(im_shape[0] * ratio)), random.randint(0, int(im_shape[1] * ratio)))
    # use this to always mask the whole image
    if mask_full_image:
        size = (int(im_shape[0] * ratio), int(im_shape[1] * ratio))
    limits = (im_shape[0] - size[0] // 2, im_shape[1] - size[1] // 2)
    center = (random.randint(size[0] // 2, limits[0]), random.randint(size[1] // 2, limits[1]))
    draw_type = random.randint(0, 1)
    if draw_type == 0 or mask_full_image:
        draw.rectangle(
            (center[0] - size[0] // 2, center[1] - size[1] // 2, center[0] + size[0] // 2, center[1] + size[1] // 2),
            fill=255,
        )
    else:
        draw.ellipse(
            (center[0] - size[0] // 2, center[1] - size[1] // 2, center[0] + size[0] // 2, center[1] + size[1] // 2),
            fill=255,
        )

    return mask


class LinearRamp:
    def __init__(self, start_value=0, end_value=1, start_iter=-1, end_iter=0):
        self.start_value = start_value
        self.end_value = end_value
        self.start_iter = start_iter
        self.end_iter = end_iter

    def __call__(self, i):
        if i < self.start_iter:
            return self.start_value
        if i >= self.end_iter:
            return self.end_value
        part = (i - self.start_iter) / (self.end_iter - self.start_iter)
        return self.start_value * (1 - part) + self.end_value * part

class DrawMethod(Enum):
    LINE = 'line'
    CIRCLE = 'circle'
    SQUARE = 'square'

def load_yaml(path):
    with open(path, 'r') as f:
        return EasyDict(yaml.safe_load(f))

def make_random_irregular_mask(shape, max_angle=4, max_len=60, max_width=20, min_times=0, max_times=10,
                               draw_method=DrawMethod.LINE):
    draw_method = DrawMethod(draw_method)

    height, width = shape
    mask = np.zeros((height, width), np.float32)
    times = np.random.randint(min_times, max_times + 1)
    for i in range(times):
        start_x = np.random.randint(width)
        start_y = np.random.randint(height)
        for j in range(1 + np.random.randint(5)):
            angle = 0.01 + np.random.randint(max_angle)
            if i % 2 == 0:
                angle = 2 * 3.1415926 - angle
            length = 10 + np.random.randint(max_len)
            brush_w = 5 + np.random.randint(max_width)
            end_x = np.clip((start_x + length * np.sin(angle)).astype(np.int32), 0, width)
            end_y = np.clip((start_y + length * np.cos(angle)).astype(np.int32), 0, height)
            if draw_method == DrawMethod.LINE:
                cv2.line(mask, (start_x, start_y), (end_x, end_y), 1.0, brush_w)
            elif draw_method == DrawMethod.CIRCLE:
                cv2.circle(mask, (start_x, start_y), radius=brush_w, color=1., thickness=-1)
            elif draw_method == DrawMethod.SQUARE:
                radius = brush_w // 2
                mask[start_y - radius:start_y + radius, start_x - radius:start_x + radius] = 1
            start_x, start_y = end_x, end_y
    return mask[None, ...]


class RandomIrregularMaskGenerator:
    def __init__(self, max_angle=4, max_len=60, max_width=20, min_times=0, max_times=10, ramp_kwargs=None,
                 draw_method=DrawMethod.LINE):
        self.max_angle = max_angle
        self.max_len = max_len
        self.max_width = max_width
        self.min_times = min_times
        self.max_times = max_times
        self.draw_method = draw_method
        self.ramp = LinearRamp(**ramp_kwargs) if ramp_kwargs is not None else None

    def __call__(self, img, iter_i=None, raw_image=None):
        coef = self.ramp(iter_i) if (self.ramp is not None) and (iter_i is not None) else 1
        cur_max_len = int(max(1, self.max_len * coef))
        cur_max_width = int(max(1, self.max_width * coef))
        cur_max_times = int(self.min_times + 1 + (self.max_times - self.min_times) * coef)
        return make_random_irregular_mask(img.shape[1:], max_angle=self.max_angle, max_len=cur_max_len,
                                          max_width=cur_max_width, min_times=self.min_times, max_times=cur_max_times,
                                          draw_method=self.draw_method)


def make_random_rectangle_mask(shape, margin=10, bbox_min_size=30, bbox_max_size=100, min_times=0, max_times=3):
    height, width = shape
    mask = np.zeros((height, width), np.float32)
    bbox_max_size = min(bbox_max_size, height - margin * 2, width - margin * 2)
    times = np.random.randint(min_times, max_times + 1)
    for i in range(times):
        box_width = np.random.randint(bbox_min_size, bbox_max_size)
        box_height = np.random.randint(bbox_min_size, bbox_max_size)
        start_x = np.random.randint(margin, width - margin - box_width + 1)
        start_y = np.random.randint(margin, height - margin - box_height + 1)
        mask[start_y:start_y + box_height, start_x:start_x + box_width] = 1
    return mask[None, ...]


class RandomRectangleMaskGenerator:
    def __init__(self, margin=10, bbox_min_size=30, bbox_max_size=100, min_times=0, max_times=3, ramp_kwargs=None):
        self.margin = margin
        self.bbox_min_size = bbox_min_size
        self.bbox_max_size = bbox_max_size
        self.min_times = min_times
        self.max_times = max_times
        self.ramp = LinearRamp(**ramp_kwargs) if ramp_kwargs is not None else None

    def __call__(self, img, iter_i=None, raw_image=None):
        coef = self.ramp(iter_i) if (self.ramp is not None) and (iter_i is not None) else 1
        cur_bbox_max_size = int(self.bbox_min_size + 1 + (self.bbox_max_size - self.bbox_min_size) * coef)
        cur_max_times = int(self.min_times + (self.max_times - self.min_times) * coef)
        return make_random_rectangle_mask(img.shape[1:], margin=self.margin, bbox_min_size=self.bbox_min_size,
                                          bbox_max_size=cur_bbox_max_size, min_times=self.min_times,
                                          max_times=cur_max_times)

class MixedMaskGenerator:
    def __init__(self, irregular_proba=0, irregular_kwargs=None,
                 box_proba=0, box_kwargs=None,
                 segm_proba=0, segm_kwargs=None,
                 squares_proba=0, squares_kwargs=None,
                 superres_proba=0, superres_kwargs=None,
                 outpainting_proba=0, outpainting_kwargs=None,
                 invert_proba=0):
        self.probas = []
        self.gens = []

        if irregular_proba > 0:
            self.probas.append(irregular_proba)
            if irregular_kwargs is None:
                irregular_kwargs = {}
            else:
                irregular_kwargs = dict(irregular_kwargs)
            irregular_kwargs['draw_method'] = DrawMethod.LINE
            self.gens.append(RandomIrregularMaskGenerator(**irregular_kwargs))

        if box_proba > 0:
            self.probas.append(box_proba)
            if box_kwargs is None:
                box_kwargs = {}
            self.gens.append(RandomRectangleMaskGenerator(**box_kwargs))

        if squares_proba > 0:
            self.probas.append(squares_proba)
            if squares_kwargs is None:
                squares_kwargs = {}
            else:
                squares_kwargs = dict(squares_kwargs)
            squares_kwargs['draw_method'] = DrawMethod.SQUARE
            self.gens.append(RandomIrregularMaskGenerator(**squares_kwargs))

        self.probas = np.array(self.probas, dtype='float32')
        self.probas /= self.probas.sum()
        self.invert_proba = invert_proba

    def __call__(self, img, iter_i=None, raw_image=None):
        kind = np.random.choice(len(self.probas), p=self.probas)
        gen = self.gens[kind]
        result = gen(img, iter_i=iter_i, raw_image=raw_image)
        if self.invert_proba > 0 and random.random() < self.invert_proba:
            result = 1 - result
        return result
    
class LaMaMaskGenerator:
    def __init__(self,config_path):
        config = load_yaml(config_path)
        self.mask_generator = MixedMaskGenerator(**config.mask_generator_kwargs)

    def __call__(self, src_image):
        if type(src_image) != np.ndarray:
            src_image = np.array(src_image)
        img = np.transpose(src_image, (2, 0, 1))
        src_mask = self.mask_generator(img)[0]
        mask = np.clip(src_mask * 255, 0, 255).astype('uint8')
        return mask
    
