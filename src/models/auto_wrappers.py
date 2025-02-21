import copy
import os

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForCTC,
    AutoModelForPreTraining,
    PretrainedConfig,
)
from transformers.dynamic_module_utils import (
    get_class_from_dynamic_module,
    resolve_trust_remote_code,
)
from transformers.models.auto.auto_factory import _BaseAutoModelClass, _get_model_class

from models.embeddings import AdaptiveEmbedding, PositionalEmbedding
from models.extractors import Conv2dFeatureExtractor
from models.streaming_modules import FeatureExtractorForStreaming


class FeatureExtractionInitModifier(type):
    def __new__(cls, name, bases, dct):
        # Create the class using the original definition
        new_cls = super().__new__(cls, name, bases, dct)

        # Save the original __init__ method
        original_init = new_cls.__init__

        # Modify the __init__ method dynamically
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if hasattr(self.config, "num_mel_bins"):
                if not hasattr(self.config, "expect_2d_input"):
                    self.config.expect_2d_input = True
                    self.config.second_dim_input_size = self.config.num_mel_bins
            if hasattr(self.config, "expect_2d_input") and self.config.expect_2d_input:
                getattr(self, self.base_model_prefix).feature_extractor = Conv2dFeatureExtractor(self.config)

        # Replace the __init__ method with the modified version
        new_cls.__init__ = new_init

        return new_cls


class CustomAutoModelWrapper(_BaseAutoModelClass):
    @staticmethod
    def get_streaming_model_class(original_cls, config):
        class StreamingModel(FeatureExtractorForStreaming, original_cls):
            pass

        if hasattr(config, "is_causal"):
            if config.is_causal and config.expect_2d_input:
                return StreamingModel
            elif config.is_causal:
                raise NotImplementedError("Causal streaming models are not supported for 1d input")

        return original_cls

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        kwargs["_from_auto"] = True
        hub_kwargs_names = [
            "cache_dir",
            "code_revision",
            "force_download",
            "local_files_only",
            "proxies",
            "resume_download",
            "revision",
            "subfolder",
            "use_auth_token",
        ]
        hub_kwargs = {name: kwargs.pop(name) for name in hub_kwargs_names if name in kwargs}
        if not isinstance(config, PretrainedConfig):
            kwargs_orig = copy.deepcopy(kwargs)
            # ensure not to pollute the config object with torch_dtype="auto" - since it's
            # meaningless in the context of the config object - torch.dtype values are acceptable
            if kwargs.get("torch_dtype", None) == "auto":
                _ = kwargs.pop("torch_dtype")

            config, kwargs = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                return_unused_kwargs=True,
                trust_remote_code=trust_remote_code,
                **hub_kwargs,
                **kwargs,
            )

            # if torch_dtype=auto was passed here, ensure to pass it on
            if kwargs_orig.get("torch_dtype", None) == "auto":
                kwargs["torch_dtype"] = "auto"

        has_remote_code = hasattr(config, "auto_map") and cls.__name__ in config.auto_map
        has_local_code = type(config) in cls._model_mapping.keys()
        trust_remote_code = resolve_trust_remote_code(
            trust_remote_code, pretrained_model_name_or_path, has_local_code, has_remote_code
        )
        if has_remote_code and trust_remote_code:
            class_ref = config.auto_map[cls.__name__]
            model_class = get_class_from_dynamic_module(
                class_ref, pretrained_model_name_or_path, **hub_kwargs, **kwargs
            )
            model_class = FeatureExtractionInitModifier(model_class.__name__, (model_class,), {})
            model_class = CustomAutoModelWrapper.get_streaming_model_class(model_class, config)
            _ = hub_kwargs.pop("code_revision", None)
            if os.path.isdir(pretrained_model_name_or_path):
                model_class.register_for_auto_class(cls.__name__)
            else:
                cls.register(config.__class__, model_class, exist_ok=True)
            return model_class.from_pretrained(
                pretrained_model_name_or_path, *model_args, config=config, **hub_kwargs, **kwargs
            )
        elif type(config) in cls._model_mapping.keys():
            model_class = _get_model_class(config, cls._model_mapping)
            model_class = FeatureExtractionInitModifier(model_class.__name__, (model_class,), {})
            model_class = CustomAutoModelWrapper.get_streaming_model_class(model_class, config)
            return model_class.from_pretrained(
                pretrained_model_name_or_path, *model_args, config=config, **hub_kwargs, **kwargs
            )
        raise ValueError(
            f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}.\n"
            f"Model type should be one of {', '.join(c.__name__ for c in cls._model_mapping.keys())}."
        )

    @classmethod
    def from_config(cls, config, **kwargs):
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        has_remote_code = hasattr(config, "auto_map") and cls.__name__ in config.auto_map
        has_local_code = type(config) in cls._model_mapping.keys()
        trust_remote_code = resolve_trust_remote_code(
            trust_remote_code, config._name_or_path, has_local_code, has_remote_code
        )

        if has_remote_code and trust_remote_code:
            class_ref = config.auto_map[cls.__name__]
            if "--" in class_ref:
                repo_id, class_ref = class_ref.split("--")
            else:
                repo_id = config.name_or_path
            model_class = get_class_from_dynamic_module(class_ref, repo_id, **kwargs)
            if os.path.isdir(config._name_or_path):
                model_class.register_for_auto_class(cls.__name__)
            else:
                cls.register(config.__class__, model_class, exist_ok=True)
            _ = kwargs.pop("code_revision", None)
            model_class = FeatureExtractionInitModifier(model_class.__name__, (model_class,), {})
            model_class = CustomAutoModelWrapper.get_streaming_model_class(model_class, config)
            return model_class._from_config(config, **kwargs)
        elif type(config) in cls._model_mapping.keys():
            model_class = _get_model_class(config, cls._model_mapping)
            model_class = FeatureExtractionInitModifier(model_class.__name__, (model_class,), {})
            model_class = CustomAutoModelWrapper.get_streaming_model_class(model_class, config)
            return model_class._from_config(config, **kwargs)

        raise ValueError(
            f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}.\n"
            f"Model type should be one of {', '.join(c.__name__ for c in cls._model_mapping.keys())}."
        )


class CustomAutoModelForCTC(CustomAutoModelWrapper, AutoModelForCTC):
    pass


class CustomAutoModelForPretraining(CustomAutoModelWrapper, AutoModelForPreTraining):
    pass


class PositionalEncodingInitModifier(type):
    def __new__(cls, name, bases, dct):
        # Create the class using the original definition
        new_cls = super().__new__(cls, name, bases, dct)

        # Save the original __init__ method
        original_init = new_cls.__init__

        # Modify the __init__ method dynamically
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if hasattr(self.config, "pos_emb_fixed") and self.config.pos_emb_fixed:
                self.transformer.wte = AdaptiveEmbedding(
                    n_token=self.config.vocab_size,
                    d_embed=self.config.hidden_size,
                    d_proj=self.config.hidden_size,
                    cutoffs=[],
                )
                self.transformer.wpe = PositionalEmbedding(demb=self.config.hidden_size)

        # Replace the __init__ method with the modified version
        new_cls.__init__ = new_init

        return new_cls


class CustomModelForCausalLM(AutoModelForCausalLM):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        kwargs["_from_auto"] = True
        hub_kwargs_names = [
            "cache_dir",
            "code_revision",
            "force_download",
            "local_files_only",
            "proxies",
            "resume_download",
            "revision",
            "subfolder",
            "use_auth_token",
        ]
        hub_kwargs = {name: kwargs.pop(name) for name in hub_kwargs_names if name in kwargs}
        if not isinstance(config, PretrainedConfig):
            kwargs_orig = copy.deepcopy(kwargs)
            # ensure not to pollute the config object with torch_dtype="auto" - since it's
            # meaningless in the context of the config object - torch.dtype values are acceptable
            if kwargs.get("torch_dtype", None) == "auto":
                _ = kwargs.pop("torch_dtype")

            config, kwargs = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                return_unused_kwargs=True,
                trust_remote_code=trust_remote_code,
                **hub_kwargs,
                **kwargs,
            )

            # if torch_dtype=auto was passed here, ensure to pass it on
            if kwargs_orig.get("torch_dtype", None) == "auto":
                kwargs["torch_dtype"] = "auto"

        has_remote_code = hasattr(config, "auto_map") and cls.__name__ in config.auto_map
        has_local_code = type(config) in cls._model_mapping.keys()
        trust_remote_code = resolve_trust_remote_code(
            trust_remote_code, pretrained_model_name_or_path, has_local_code, has_remote_code
        )
        if has_remote_code and trust_remote_code:
            class_ref = config.auto_map[cls.__name__]
            model_class = get_class_from_dynamic_module(
                class_ref, pretrained_model_name_or_path, **hub_kwargs, **kwargs
            )
            model_class = PositionalEncodingInitModifier(model_class.__name__, (model_class,), {})
            _ = hub_kwargs.pop("code_revision", None)
            if os.path.isdir(pretrained_model_name_or_path):
                model_class.register_for_auto_class(cls.__name__)
            else:
                cls.register(config.__class__, model_class, exist_ok=True)
            return model_class.from_pretrained(
                pretrained_model_name_or_path, *model_args, config=config, **hub_kwargs, **kwargs
            )
        elif type(config) in cls._model_mapping.keys():
            model_class = _get_model_class(config, cls._model_mapping)
            model_class = PositionalEncodingInitModifier(model_class.__name__, (model_class,), {})
            return model_class.from_pretrained(
                pretrained_model_name_or_path, *model_args, config=config, **hub_kwargs, **kwargs
            )
        raise ValueError(
            f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}.\n"
            f"Model type should be one of {', '.join(c.__name__ for c in cls._model_mapping.keys())}."
        )

    @classmethod
    def from_config(cls, config, **kwargs):
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        has_remote_code = hasattr(config, "auto_map") and cls.__name__ in config.auto_map
        has_local_code = type(config) in cls._model_mapping.keys()
        trust_remote_code = resolve_trust_remote_code(
            trust_remote_code, config._name_or_path, has_local_code, has_remote_code
        )

        if has_remote_code and trust_remote_code:
            class_ref = config.auto_map[cls.__name__]
            if "--" in class_ref:
                repo_id, class_ref = class_ref.split("--")
            else:
                repo_id = config.name_or_path
            model_class = get_class_from_dynamic_module(class_ref, repo_id, **kwargs)
            if os.path.isdir(config._name_or_path):
                model_class.register_for_auto_class(cls.__name__)
            else:
                cls.register(config.__class__, model_class, exist_ok=True)
            _ = kwargs.pop("code_revision", None)
            model_class = PositionalEncodingInitModifier(model_class.__name__, (model_class,), {})
            return model_class._from_config(config, **kwargs)
        elif type(config) in cls._model_mapping.keys():
            model_class = _get_model_class(config, cls._model_mapping)
            model_class = PositionalEncodingInitModifier(model_class.__name__, (model_class,), {})
            return model_class._from_config(config, **kwargs)

        raise ValueError(
            f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}.\n"
            f"Model type should be one of {', '.join(c.__name__ for c in cls._model_mapping.keys())}."
        )
