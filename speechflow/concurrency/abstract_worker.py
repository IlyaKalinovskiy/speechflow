import abc
import logging

from speechflow.logging import trace

LOGGER = logging.getLogger("root")


class AbstractWorker:
    """Basic interface for a worker class."""

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def activate(self):
        """Sets execution flag to true.

        join() should be called separately.

        """

    @abc.abstractmethod
    def deactivate(self):
        """Sets execution flag to false.

        join() should be called separately.

        """

    @abc.abstractmethod
    def is_active(self) -> bool:
        """Return the execution flag value."""
        pass

    @abc.abstractmethod
    def started(self):
        """Sets execution flag to true."""
        pass

    @abc.abstractmethod
    def finished(self):
        """Sets execution flag to false.

        join() should be called separately.

        """

    @abc.abstractmethod
    def is_started(self) -> bool:
        """Return the execution flag value."""
        pass

    @abc.abstractmethod
    def do_work_once(self):
        """Implement a single iteration of the main loop here."""
        pass

    def on_start(self):
        """Implement in a derivative class, if you wish to do something before the main
        loop."""
        pass

    def on_crash(self):
        """Implement in a derivative class, if you wish to perform some operations in the
        case of an unhandled exception.

        You may use sys.exc_info() to get stacktrace.

        """
        pass

    def on_finish(self):
        """Implement in a derivative class, if you wish to do something."""
        pass

    def _run(self):
        """This method should be run in the new thread/process.

        In case of standard threads/processes, it should be called from run()
        function.

        """
        try:
            self.on_start()
            self.started()
        except Exception as e:
            LOGGER.error(trace(self, e, message="on_start thrown an exception"))
            raise e

        try:
            while self.is_active() and self.is_started():
                try:
                    self.do_work_once()
                except KeyboardInterrupt:
                    LOGGER.error(trace(self, "interrupt received, stopping ..."))
                    break

        except Exception as e:
            LOGGER.error(trace(self, e, message="do_work_once thrown an exception"))
            try:
                self.on_crash()
            except Exception as e:
                LOGGER.error(trace(self, e, message="on_crash thrown an exception"))
                raise e

        finally:
            try:
                self.on_finish()
            except Exception as e:
                LOGGER.error(trace(self, e, message="on_finish thrown an exception"))
                raise e

            finally:
                self.finished()
