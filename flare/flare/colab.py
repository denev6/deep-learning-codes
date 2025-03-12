import sys
import os

class GoogleDrive:
    """Mount Google Drive for Colab env

    :param mount_path: path to mount Google Drive (default: "/content/drive")
    :param project_path: path to look for your files in Google Drive (default: "/content/drive/MyDrive")
    :param force_remount: force remount Google Drive (default: False)
    """
    def __init__(self, mount_path="/content/drive", project_path="/content/drive/MyDrive", force_remount=False):
        from google.colab import drive
        drive.mount(mount_path, force_remount=force_remount)

        self._project_path = project_path
        sys.path.append(self._project_path)

    def join(self, *args):
        """Join Google Drive path"""
        return os.path.join(self._project_path, *args)
