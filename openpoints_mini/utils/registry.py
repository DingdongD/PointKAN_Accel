# Simplified from OpenPoints / mmcv-style Registry (no mmcv dependency).
import copy
import inspect
import warnings
from functools import partial


def _is_seq_of(seq, elem_type):
    return isinstance(seq, (list, tuple)) and all(isinstance(x, elem_type) for x in seq)


class Registry:
    def __init__(self, name, build_func=None, parent=None, scope=None):
        self._name = name
        self._module_dict = dict()
        self._children = dict()
        self._scope = self.infer_scope() if scope is None else scope
        if build_func is None:
            if parent is not None:
                self.build_func = parent.build_func
            else:
                self.build_func = build_from_cfg
        else:
            self.build_func = build_func
        if parent is not None:
            assert isinstance(parent, Registry)
            parent._add_children(self)
            self.parent = parent
        else:
            self.parent = None

    @staticmethod
    def infer_scope():
        filename = inspect.getmodule(inspect.stack()[2][0]).__name__
        return filename.split(".")[0]

    @staticmethod
    def split_scope_key(key):
        split_index = key.find(".")
        if split_index != -1:
            return key[:split_index], key[split_index + 1 :]
        return None, key

    @property
    def name(self):
        return self._name

    @property
    def scope(self):
        return self._scope

    def get(self, key):
        scope, real_key = self.split_scope_key(key)
        if scope is None or scope == self._scope:
            if real_key in self._module_dict:
                return self._module_dict[real_key]
        elif scope in self._children:
            return self._children[scope].get(real_key)
        else:
            parent = self.parent
            while parent is not None and parent.parent is not None:
                parent = parent.parent
            if parent is not None:
                return parent.get(key)
        return None

    def build(self, *args, **kwargs):
        return self.build_func(*args, **kwargs, registry=self)

    def _add_children(self, registry):
        assert registry.scope not in self.children
        self._children[registry.scope] = registry

    @property
    def children(self):
        return self._children

    def _register_module(self, module_class, module_name=None, force=False):
        if not inspect.isclass(module_class):
            raise TypeError("module must be a class")
        if module_name is None:
            module_name = module_class.__name__
        if isinstance(module_name, str):
            module_name = [module_name]
        for name in module_name:
            if not force and name in self._module_dict:
                raise KeyError(f"{name} is already registered in {self.name}")
            self._module_dict[name] = module_class

    def deprecated_register_module(self, cls=None, force=False):
        warnings.warn("deprecated_register_module is deprecated", DeprecationWarning, stacklevel=2)
        if cls is None:
            return partial(self.deprecated_register_module, force=force)
        self._register_module(cls, force=force)
        return cls

    def register_module(self, name=None, force=False, module=None):
        if not isinstance(force, bool):
            raise TypeError("force must be a boolean")
        if isinstance(name, type):
            return self.deprecated_register_module(name, force=force)
        if not (name is None or isinstance(name, str) or _is_seq_of(name, str)):
            raise TypeError("name must be None, str, or sequence of str")
        if module is not None:
            self._register_module(module_class=module, module_name=name, force=force)
            return module

        def _register(cls):
            self._register_module(module_class=cls, module_name=name, force=force)
            return cls

        return _register


def build_from_cfg(cfg, registry, default_args=None):
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a dict, got {type(cfg)}")
    if "NAME" not in cfg:
        if default_args is None or "NAME" not in default_args:
            raise KeyError('cfg must contain "NAME"')
    if not isinstance(registry, Registry):
        raise TypeError("registry must be Registry")
    obj_type = cfg.get("NAME")
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(f"{obj_type} is not in registry {registry.name}")
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(f"NAME must be str or class, got {type(obj_type)}")
    obj_cfg = copy.deepcopy(cfg)
    if default_args is not None:
        obj_cfg.update(default_args)
    obj_cfg.pop("NAME")
    try:
        return obj_cls(**obj_cfg)
    except Exception as e:
        raise type(e)(f"{obj_cls.__name__}: {e}") from e
