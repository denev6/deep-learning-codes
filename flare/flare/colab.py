import sys
import os
import importlib.util


class Colab:
    """Mount Google Drive for Colab enviroment

    Args:
        mount_path: path to mount Google Drive (default: "/content/drive")
        project_path: path to look for your files in Google Drive (default: "/content/drive/MyDrive")

    Example:
        >>> gdrive = Colab()
        >>> gdrive.mount_drive()
    """

    def __init__(
        self, mount_path="/content/drive", project_path="/content/drive/MyDrive"
    ):
        assert (
            importlib.util.find_spec("google.colab") is not None
        ), "This class is only available in Google Colab. Cannot import `google.colab` in current environment."
        self._mount_path = mount_path
        self._project_path = project_path

    def mount_drive(self, force_remount=False):
        """Mount Google Drive

        Args:
            force_remount: force remount Google Drive (default: False)
        """
        from google.colab import drive

        drive.mount(self._mount_path, force_remount=force_remount)
        sys.path.append(self._project_path)

    def join(self, *args):
        return os.path.join(self._project_path, *args)
