import io
import numpy as np
import PIL.Image as Image

try:
    import sys
    sys.path.append('/mnt/lustre/share/pymc/py3/')
    import mc
except:
    pass


class MClientReader(object):
    
    def __init__(self):
        self.initialized = False

    def _init_memcached(self):
        if not self.initialized:
            server_list_config_file = '/mnt/lustre/share/memcached_client/server_list.conf'
            client_config_file = '/mnt/lustre/share/memcached_client/client.conf'
            self.mclient = mc.MemcachedClient.GetInstance(
                server_list_config_file, client_config_file
            )
            self.initialized = True

    def open(self, image_path, binary=False):
        self._init_memcached()
        value = mc.pyvector()
        self.mclient.Get(image_path, value)
        value_buf = mc.ConvertBuffer(value)
        buff = io.BytesIO(value_buf)
        with Image.open(buff) as image:
            if binary:
                image = image.convert('L')
            else:
                image = image.convert('RGB')
        return image
