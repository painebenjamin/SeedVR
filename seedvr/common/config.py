# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

"""
Configuration utility functions
"""

import importlib
import os
from collections.abc import Callable
from functools import partial
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

OmegaConf.register_new_resolver("eval", eval)


def load_config(
    path: str, argv: list[str] = None, root_dir: str | None = None
) -> DictConfig | ListConfig:
    """
    Load a configuration. Will resolve inheritance.
    """
    config = OmegaConf.load(path)
    if argv is not None:
        config_argv = OmegaConf.from_dotlist(argv)
        config = OmegaConf.merge(config, config_argv)
    resolve_inheritance_fn = partial(resolve_inheritance, root_dir=root_dir)
    config = resolve_recursive(config, resolve_inheritance_fn)
    return config


def resolve_recursive(
    config: Any,
    resolver: Callable[[DictConfig | ListConfig], DictConfig | ListConfig],
) -> Any:
    config = resolver(config)
    if isinstance(config, DictConfig):
        for k in config.keys():
            v = config.get(k)
            if isinstance(v, (DictConfig, ListConfig)):
                config[k] = resolve_recursive(v, resolver)
    if isinstance(config, ListConfig):
        for i in range(len(config)):
            v = config.get(i)
            if isinstance(v, (DictConfig, ListConfig)):
                config[i] = resolve_recursive(v, resolver)
    return config


def resolve_inheritance(
    config: DictConfig | ListConfig, root_dir: str | None = None
) -> Any:
    """
    Recursively resolve inheritance if the config contains:
    __inherit__: path/to/parent.yaml or a ListConfig of such paths.
    """
    if isinstance(config, DictConfig):
        inherit = config.pop("__inherit__", None)

        if inherit:
            inherit_list = inherit if isinstance(inherit, ListConfig) else [inherit]

            parent_config = None
            for parent_path in inherit_list:
                assert isinstance(parent_path, str)
                if root_dir is not None:
                    parent_path = os.path.join(root_dir, parent_path)

                parent_config = (
                    load_config(parent_path)
                    if parent_config is None
                    else OmegaConf.merge(parent_config, load_config(parent_path))
                )

            if len(config.keys()) > 0:
                config = OmegaConf.merge(parent_config, config)
            else:
                config = parent_config
    return config


def import_item(path: str, name: str) -> Any:
    """
    Import a python item. Example: import_item("path.to.file", "MyClass") -> MyClass
    """
    return getattr(importlib.import_module(path), name)


def create_object(config: DictConfig) -> Any:
    """
    Create an object from config.
    The config is expected to contains the following:
    __object__:
      path: path.to.module
      name: MyClass
      args: as_config | as_params (default to as_config)
    """
    item = import_item(
        path=config.__object__.path,
        name=config.__object__.name,
    )
    args = config.__object__.get("args", "as_config")
    if args == "as_config":
        return item(config)
    if args == "as_params":
        config = OmegaConf.to_object(config)
        config.pop("__object__")
        return item(**config)
    raise NotImplementedError(f"Unknown args type: {args}")
