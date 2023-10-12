import NuRadioReco.eventbrowser.dataprovider_root
import NuRadioReco.eventbrowser.dataprovider_nur
import threading
import logging
logging.basicConfig()
logger = logging.getLogger('eventbrowser.dataprovider')
logger.setLevel(logging.INFO)

LOCK = threading.Lock()

class DataProvider(object):
    __instance = None

    def __new__(cls):
        if DataProvider.__instance is None:
            DataProvider.__instance = object.__new__(cls)
        return DataProvider.__instance

    def __init__(self):
        self.__data_provider = None

    def set_filetype(self, use_root):
        if use_root:
            self.__data_provider = NuRadioReco.eventbrowser.dataprovider_root.DataProviderRoot()
        else:
            self.__data_provider = NuRadioReco.eventbrowser.dataprovider_nur.DataProvider()

    def get_file_handler(self, user_id, filename):
        thread = threading.get_ident()
        total_threads = threading.active_count()
        logger.debug(f"Thread {thread} out of total {total_threads} requesting file_handler for user {user_id}")
        LOCK.acquire()
        logger.debug(f"Thread {thread} locked, getting file_handler for user {user_id}...")
        file_handler = self.__data_provider.get_file_handler(user_id, filename)
        LOCK.release()
        logger.debug(f"Returning file_handler and releasing lock")
        return file_handler
