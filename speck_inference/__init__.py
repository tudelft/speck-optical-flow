import os
import platform
import shutil
import sys

module_name = '%s' % (__name__)

# The following is needed to put everything into the samna namespace. Otherwise it will end up in samna.samna.

samna_dir_path = os.path.dirname(__file__)

sys.path.insert(0, samna_dir_path)

del sys.modules[module_name]

sys.modules[module_name] = __import__("samna")

sys.modules[module_name].__dict__.update( __import__("submodules").__dict__)

# Configure the JIT on import.

jit_config_file = os.path.join(samna_dir_path, "JitConfig.json")

assert(os.path.isfile(jit_config_file)), 'JitConfig.json file missing!'

sys.modules[module_name].jit.init_jit_configuration(jit_config_file, samna_dir_path)

# Prophesee cameras are loaded as plugins with dlopen
# MV_HAL_PLUGIN_PATH defines the path where the loader looks for the plugins
os.environ['MV_HAL_PLUGIN_PATH'] = os.path.join(samna_dir_path, 'metavision/hal/plugins/')

def __install_psee_rules__():
    if platform.system() == 'Linux':
        shutil.copyfile(os.path.join(samna_dir_path, 'metavision/rules/88-cyusb.rules'), '/etc/udev/rules.d/88-cyusb.rules')
        shutil.copyfile(os.path.join(samna_dir_path, 'metavision/rules/99-evkv2.rules'), '/etc/udev/rules.d/99-evkv2.rules')
    else:
        raise OSError('PSEE USB rules can be installed only in Linux!')

sys.modules[module_name].__install_psee_rules__ = __install_psee_rules__

from pathlib import Path

from sys import implementation as impl

cache_tag = f"{impl.name}-{impl.version.major}.{impl.version.minor}.{impl.version.micro}-samna.{sys.modules[module_name].__version__}"

sys.modules[module_name].jit.set_jit_cache_path(str(Path.home() / ".samna" / "jit_cache" / cache_tag))

# The following functions are needed only on module setup and should not be called by the user.

del sys.modules[module_name].jit.set_jit_cache_path

del sys.modules[module_name].jit.init_jit_configuration

del samna_dir_path, jit_config_file, module_name, Path, impl, cache_tag
