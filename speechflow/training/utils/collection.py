import typing as tp

__all__ = ["ComponentCollection", "ModelCollection"]


class ComponentCollection:
    components: tp.Dict[str, object]

    def __init__(self):
        self.components: tp.Dict[str, tp.Union[object, tp.Tuple[object, object]]] = {}

    def _check(self, component_name: str):
        if component_name in self.components:
            raise RuntimeError(f"Component '{component_name}' already registered")

    def registry_module(self, module, filter_names=None):
        keys = module.__dict__.keys()
        if filter_names is not None:
            keys = [k for k in keys if filter_names(k)]

        for key in keys:
            if f"{key}Params" in module.__dict__.keys():
                self._check(key)
                self.components[key] = (
                    module.__dict__[key],
                    module.__dict__[f"{key}Params"],
                )

    def registry_component(self, component: object, component_params: object = None):
        component_name = component.__name__
        self._check(component_name)
        if component_params is not None:
            self.components[component_name] = (component, component_params)
        else:
            self.components[component_name] = component

    def __contains__(self, key):
        return key in self.components

    def __getitem__(self, item: str) -> tp.Union[object, tp.Tuple[object, object]]:
        if item not in self.components:
            raise RuntimeError(f"Component '{item}' not found")

        return self.components[item]


class ModelCollection:
    models: tp.Dict[str, object]

    def __init__(self):
        self.models: tp.Dict[str, tp.Union[object, tp.Tuple[object, object]]] = {}

    def _check(self, model_name: str):
        if model_name in self.models:
            raise RuntimeError(f"Model '{model_name}' already registered")

    def registry_model(
        self, model: object, model_params: object = None, tag: tp.Optional[str] = None
    ):
        model_name = model.__name__ if tag is None else tag
        self._check(model_name)
        if model_params is not None:
            self.models[model_name] = (model, model_params)
        else:
            self.models[model_name] = model

    def __contains__(self, key):
        return key in self.models

    def __getitem__(self, item: str) -> tp.Union[object, tp.Tuple[object, object]]:
        if item not in self.models:
            raise RuntimeError(f"Model '{item}' not found")

        return self.models[item]
