"""
Argument parsers
"""

import re
import copy
import logging
import functools
from inspect import getfullargspec
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Set, Tuple, Type, Union, Dict, Callable
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, _ArgumentGroup
from munch import Munch

log = logging.getLogger(__name__)


class BaseConfParser(ArgumentParser):
    """Base class of all argument parses"""

    argparse_tags: Tuple[str, str]
    """Set by the `Tagged` wrapper"""

    def __init__(self, *args, **kwargs):
        self._deps = {}
        if 'formatter_class' not in kwargs:
            kwargs['formatter_class'] = ArgumentDefaultsHelpFormatter
        if 'description' not in kwargs:
            kwargs['description'] = "About config groups: (S): Configs are " \
                "passed to and only used by the server. (S->C): Configs are " \
                "passed to sever, then send/broadcast to clients by the " \
                "server and used by the clients. (C): Configs are passed to " \
                "and only used by the client(s)."
        super().__init__(*args, **kwargs)

    def add_argument_group(self, title: str, *args, **kwargs):
        return super().add_argument_group(
            f"\033[1m{title}\033[0m", *args, **kwargs)

    def register_cfg_dep(
            self, key: str, depends_on: Union[str, Callable[[Munch], bool]]):
        """
        Make config entry `key` depends on a predicate or another key.
        If that predicate evaluates to false or the other key does not exist
        in parsed config, `key` will be removed from the parsed config.
        The predicate takes the parse config as input, and might be evaluated
        multiple times. (Configs are sanitized iteratively)
        """
        dest_var = key.strip('-').replace('-', '_').replace('/', '.')
        if isinstance(depends_on, str):
            dep_var = depends_on.strip('-').replace('-', '_').replace('/', '.')
            self._deps[dest_var] = lambda cfg, dep=dep_var: \
                r_hasattr(cfg, dep) and r_getattr(cfg, dep) is not None
        elif callable(depends_on):
            self._deps[dest_var] = depends_on
        else:
            raise TypeError("'depends_on' must be a config key or a callable")

    def parse_args(self, *args, **kwargs) -> Munch:
        """Parse arguments and make them hierarchical."""
        parsed_args, unknown = super().parse_known_args(*args, **kwargs)
        if len(unknown):
            log.warning("The following args are not used: %s", str(unknown))

        # pose_process args
        cfg = self.munchify_args(vars(parsed_args))
        return self.sanitize_config(cfg)

    def parse_known_args(self, *args, **kwargs) -> Tuple[Munch, List]:
        """Parse arguments and make them hierarchical."""
        parsed_args, unknown = super().parse_known_args(*args, **kwargs)
        cfg = self.munchify_args(vars(parsed_args))
        return self.sanitize_config(cfg), unknown

    def sanitize_config(self, cfg: Munch) -> Munch:
        """Remove unnecessary configs"""
        cfg_copy = copy.deepcopy(cfg)

        # Returns the number of keys removed
        to_remove = []
        def _sanitize_keys(
                cur_cfg: Union[Dict, Munch], nsp: str, full_nsp: str) -> None:
            full_nsp = f"{full_nsp}.{nsp}" if len(full_nsp) > 0 else f"{nsp}"
            if full_nsp in self._deps and not self._deps[full_nsp](cfg_copy):
                # Deferred remove to avoid changing dict during recursion
                to_remove.append((cur_cfg, nsp))
                return
            if isinstance(cur_cfg[nsp], Munch):
                # Don't remove the top-level dict, otherwise infinite loop.
                if len(cur_cfg[nsp]) == 0 and nsp != '':
                    to_remove.append((cur_cfg, nsp))
                for sub_nsp in cur_cfg[nsp].keys():
                    _sanitize_keys(cur_cfg[nsp], sub_nsp, full_nsp)

        # Sanitize keys until the config cannot be reduced anymore
        _sanitize_keys({"": cfg_copy}, "", "")
        while len(to_remove) > 0:
            for sub_cfg, key in to_remove:
                sub_cfg.pop(key)
            to_remove.clear()
            _sanitize_keys({"": cfg_copy}, "", "")

        return cfg_copy

    @staticmethod
    def munchify_args(parsed_args: Dict[str, Any]) -> Munch:
        """Make args hierarchical and check for conflict keys/namespaces"""
        cfg = Munch()
        for key, value in parsed_args.items():
            cur_cfg = cfg
            nsps = re.split(r'[\./]', key)
            for i, nsp in enumerate(nsps):
                if nsp not in cur_cfg:
                    cur_cfg[nsp] = value if i == len(nsps) - 1 else Munch()
                elif (i == len(nsps) - 1) == (isinstance(cur_cfg[nsp], Munch)):
                    raise RuntimeError(
                        f"'{'.'.join(nsps[:i+1])}' already used as a "
                        f"{'namespace' if i == len(nsps) - 1 else 'key'} "
                        "and cannot be used as a "
                        f"{'namespace' if i != len(nsps) - 1 else 'key'}")
                cur_cfg = cur_cfg[nsp]
        return cfg

    @staticmethod
    def get_value(cfg: Munch, key: str) -> Any:
        """Get value by flattened key (e.g. a/b.c). Return None if not exist"""
        return r_getattr(cfg, key.replace('/', '.'))

    @staticmethod
    def flatten(cfg: Munch):
        """
        Returns a generator that allows to iter through the flattened config.
        Example:
            # keys are joined by '.', e.g. ('training.epochs')
            for key, value in flatten(cfg):
                print(key, value)
        """
        def recursion(cfg: Munch, nsp: str):
            for cur_ns, value in cfg.items():
                full_path = f"{nsp}.{cur_ns}" if nsp != "" else f"{cur_ns}"
                if isinstance(value, Munch):
                    yield from recursion(value, full_path)
                else:
                    yield full_path, value
        return recursion(cfg, "")

    @staticmethod
    def merge(old: Munch, new: Munch) -> Munch:
        """Merge new config into old config, replacing duplicates"""
        return BaseConfParser.munchify_args({
            **dict(BaseConfParser.flatten(old)),
            **dict(BaseConfParser.flatten(new)),
        })


Conf = BaseConfParser.munchify_args
"""An alias to help construct configs directly"""


class FLConfParser(BaseConfParser):
    """Parse basic federated learning configs for both clients and servers"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        fl_args = self.add_argument_group(
            "FL Training and Communication Configs (S,S->C)")
        fl_args.add_argument(
            '-fl.sd', '--federation.seed', type=int, default=0, metavar='SEED',
            help="random initial seed")
        fl_args.add_argument(
            '-fl.te', '--federation.test-every',
            type=int, default=1, metavar='X',
            help="Test model every X communication round.")


class TrainingConfParser(BaseConfParser):
    """Parse hyperparameters for training"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Client-only training config
        training_args = self.add_argument_group(
            "Training Configs (S,S->C)")

        training_args.add_argument(
            '-tr.e', '--training.epochs', default=300, type=int, metavar='E',
            help="Maximum number of epochs to train")
        training_args.add_argument(
            '-tr.lr', '--training.learning-rate',
            default=0.1, type=float, metavar='LR',
            help="Initial lr. Also the target lr rate of warmup epochs.")
        training_args.add_argument(
            '-tr.bs', '--training.batch-size',
            default=128, type=int, metavar='SIZE',
            help="Batch size")


class EncryptionConfParser(BaseConfParser):
    """Parse arguments for the encryption module"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        enc_args = self.add_argument_group(
            "Encryption-related Configs (S,S->C)")
        enc_args.add_argument(
            '-enc.sam', '--encryption.secure-aggregation-method',
            default=None, choices=[None, 'SS', 'HE', 'SMC'],
            help="Method for secure aggregation (currently only SS supported)"
            "\n  SS: Secret Sharing"
            "\n  HE: Homomorphic Encryption"
            "\n  SMC: Secure Multiparty Computation")
        enc_args.add_argument(
            '-enc.ssn', '--encryption.secret_split_num',
            default=2, type=int, metavar='SSN',
            help="Number of splits when using the SS aggregation method.")


class QuantizationConfParser(BaseConfParser):
    """Parse arguments for the quantization module"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        c_quantize_args = self.add_argument_group(
            "Quantization-related Arguments (S,S->C)")

        c_quantize_args.add_argument(
            '-qt.q', '--quantization.quantize', action='store_true',
            help="Use quantization on gradients")
        c_quantize_args.add_argument(
            '-qt.qb', '--quantization.quantize-bits',
            default=2, type=int, metavar='BITS',
            help="Quantization bits")


class HWConfParser(BaseConfParser):
    """Parse hardwar configurations"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        hw_args = self.add_argument_group(
            "Hardware Configs (S,C)")

        hw_args.add_argument(
            '-hw.gs', '--hardware.gpus',
            default=[0], type=int, nargs='+', metavar='INDEX',
            help="Specify a list of gpu to use (INDEX starting from 0). "
            "Specify an empty list means not using GPU. "
            "(currently only 1 gpu per server/client supported)")


class DBGConfParser(BaseConfParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dbg_args = self.add_argument_group(
            "Debug Configs (S,C)")

        dbg_args.add_argument(
            '-dbg.cro', '--debug.check_required_options',
            action='store_true',
            help="Check whether requried options of each module is provided")


@dataclass
class ArgParseOption:
    """Options to pass to argparse"""
    flags: Tuple[str, str]
    """("short-option-name", "long-option-name")"""
    args: Dict[str, Any]
    """A list of arguments to feed to argparse"""

    def __init__(self, short_flag, long_flag, **kwargs) -> None:
        self.flags = (short_flag, long_flag)
        self.args = kwargs


def options(*opts: Union[str, ArgParseOption]):
    """
    A decorator to add options to class.
    Example
        @Options(
            ArgParseOption('-a','--aa', ...),
            "Group B Description",
            ArgParseOption('-b.c', '--bb.cc', ...),
            ArgParseOption('-b.d', '--bb.dd', ...))
        class Name(...):
            ...
    """
    def add_option_entries(cls: Type) -> Type:
        def argparse_options_and_groups(cls_) \
                -> Sequence[Union[str, ArgParseOption]]:
            """Return the list of argparse options associted to cls"""
            return opts if cls == cls_ else []

        def argparse_options(cls_) -> Sequence[ArgParseOption]:
            return [
                o for o in cls_.argparse_options_and_groups()
                if isinstance(o, ArgParseOption)]

        def argparse_groups(cls_) -> Sequence[str]:
            return [
                o for o in cls_.argparse_options_and_groups()
                if isinstance(o, str)]

        def add_options_to(
                cls_, arg_parser: Union[_ArgumentGroup, BaseConfParser],
                pfxs: Optional[Tuple[str, str]] = None,
                ifxs: Optional[Tuple[str, str]] = None,
                tags: Optional[Tuple[str, str]] = None) -> None:
            """
            Add options to arg_parser.

            Args:
                arg_parser: the argument parser to add options to
                pfxs: prefixes, ("s-pfx", "long-prefix"), optional
                ifxs: infixes after tag, ("s-ifx", "long-infiex"), optional
                tags: tags, ("s-tg", "long-tag"), optional

            The composed option will be:
                -s-pfx/s-tg.s-flg, --long-prefix/long-tag.long-flag

            These args can be accessed like:
                cfg.long_prefix.long_tag.flag
            """

            _a = r'[a-zA-Z]'
            _w = r'[a-zA-Z0-9]'
            _re = re.compile(
                fr'({_a}{_w}*(\-{_w}+)*(\.{_a}{_w}*(\-{_w}+)*)*)?')

            opts = cls_.argparse_options_and_groups()
            if not opts:
                return

            pfxs = ('', '') if not pfxs else pfxs
            tags = ('', '') if not tags else tags
            ifxs = ('', '') if not ifxs else ifxs

            assert len(pfxs) == 2 and all(_re.fullmatch(p) for p in pfxs), pfxs
            assert len(tags) == 2 and all(_re.fullmatch(t) for t in tags), tags
            assert len(ifxs) == 2 and all(_re.fullmatch(i) for i in ifxs), ifxs

            group = arg_parser
            for option in opts:
                if isinstance(option, str):
                    group = arg_parser.add_argument_group(option)
                    continue

                assert len(option.flags) == 2, option
                assert all(_re.fullmatch(flag)
                           for flag in option.flags), option
                group.add_argument(*[
                    f"{dash}{pfx}"
                    f"{'/' if (pfx) and tag else ''}{tag}"
                    f"{'.' if (pfx or tag) and ifx else ''}{ifx}"
                    f"{'.' if (pfx or tag or ifx) and flag else ''}{flag}"
                    for dash, pfx, tag, ifx, flag in
                    zip(('-', '--'), pfxs, tags, ifxs, option.flags)
                ], **option.args)

        cls.argparse_options_and_groups = classmethod(
            argparse_options_and_groups)
        cls.argparse_options = classmethod(argparse_options)
        cls.argparse_groups = classmethod(argparse_groups)
        cls.add_options_to = classmethod(add_options_to)
        return cls
    return add_option_entries


def require(*required_keys: str):
    """
    A decorator to examine detect arugment requirements.

    Class must have an __init__ method having the form of:
        def __init__(self, cfg:Munch, ...):
            ...

    Example:
        @Requires('cfg.a', 'cfg.b.c')
        class Name(...):
            ...
    """
    requirements: Set[str] = set(
        key.strip('-').replace('-', '_')
        for key in required_keys) or set()

    def add_requirement(cls: Type) -> Type:
        cls.cfg_required_keys = requirements.union(
            cls.cfg_required_keys
            if hasattr(cls, 'cfg_required_keys') else set())
        original_init = cls.__init__

        def new_init(_self, cfg, *args, **kwargs):
            if (r_hasattr(cfg, 'debug.check_requried_options')
                    and cfg.debug.check_required_options):
                for key in _self.cfg_required_keys:
                    assert r_hasattr(cfg, key), f"Missing key: {key}"
            original_init(_self, cfg, *args, **kwargs)

        cls.__init__ = new_init
        return cls

    return add_requirement


class Tagged:  # pylint: disable=too-few-public-methods
    """
    Tagged config parser

    Use `Tagged[ConfParser, ("s-tag", "long-tag")]` or
        `Tagged[ConfParser, "tag"]` to get the tagged class

    The tagged class have an attribute:
        .argparse_tags = ("s-tag", "long-tag")

    Modifies https://stackoverflow.com/questions/66391407
    """
    types = {}
    stags = {}

    @classmethod
    def __class_getitem__(
            cls, key: Tuple[Type[BaseConfParser], Union[Tuple[str, str], str]]):
        cls_to_tag, tags = key
        s_tag, l_tag = tags if isinstance(tags, tuple) else (tags, tags)

        kwonlyargs = getfullargspec(cls_to_tag.__init__).kwonlyargs
        assert 'stag' in kwonlyargs and 'ltag' in kwonlyargs, (
            "Tagged class must have 'stag' and 'ltag' in the kwonlyargs "
            "of its `__init__` method signature")

        name = f'{Tagged.__name__}<{cls_to_tag.__name__}, {l_tag}>'
        if name in cls.types:
            assert cls.stags[name] == s_tag, (
                f"Two tagged class having the same long tag ({l_tag}) but "
                f"different short tags ({s_tag} and {cls.stags[name]}) found.")
            return cls.types[name]

        new_type = type(name, (cls_to_tag,), {
            '__init__': functools.partialmethod(
                cls_to_tag.__init__, stag=f'/{s_tag}', ltag=f'/{l_tag}')})
        cls.types[name] = new_type
        cls.stags[name] = s_tag
        return new_type

    def __init__(self):
        raise Exception('Tagged is a static util and cannot be instantiated.')


def r_getattr(obj, attr, *args):
    """
    Get attribute recursively
    Modifies https://stackoverflow.com/questions/31174295
    """
    def _getattr(_obj, _attr):
        return getattr(_obj, _attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def r_hasattr(obj, attr):
    """Test the existence of attribute recursively"""
    return r_getattr(obj, attr, None) is not None
