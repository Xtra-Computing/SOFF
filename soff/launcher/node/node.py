"""Simple server launcher"""

from .base import BaseLauncher, BaseLauncherConfParser, alg_name_map


class NodeLauncher(BaseLauncher):
    """Simple server launcher"""

    @classmethod
    def start(cls, launcher_conf_parser_class=BaseLauncherConfParser):
        """Start the launcher"""

        # First pass to get the algorithm
        cfg, _ = cls.parse_launcher_args(launcher_conf_parser_class)

        # Insantiate class
        launcher = cls(cfg)

        # Second pass to get corresponding config
        class MergedConfParser(
                alg_name_map[cfg.launcher.algorithm].conf_parser(),
                launcher_conf_parser_class):
            """A dynamic class"""
        cfg = MergedConfParser().parse_args()

        # Launch training
        launcher.start_training(cfg)

    def start_training(self, cfg):
        """Create and start the federation server"""

        # Create server
        node = alg_name_map[self.algorithm](cfg)

        # Start training
        try:
            node.start_training(cfg)
        except Exception as exc:  # pylint: disable=broad-except
            node.log.exception(
                f"Exception from {self.__class__.__name__} captured.", exc_info=exc)
        finally:
            node.cleanup()


if __name__ == "__main__":
    NodeLauncher.start()
