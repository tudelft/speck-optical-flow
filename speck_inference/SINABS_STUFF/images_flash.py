from samna.flasher import *

# d = get_programmable_devices()
# print(d)

# # print fx3 firmware file header info
# print_firmware_header_from_file(PATH_TO_THE_FX3_IMAGE, FileType.fx3Firmware)


# candidates = [
#     "motherBoardV2_0_11_5.img",
#     "motherBoardV2_0_9_9.img",
# ]

# for p in candidates:
#     print("\n---", p)
#     try:
#         print_firmware_header_from_file(p)  # shows PRODUCT_ID, CHIP_ID, version
#     except Exception as e:
#         print("No/invalid header:", e)

from pathlib import Path
from samna.flasher import *
from time import sleep

FX3  = str(Path("motherBoardV2_0_11_5.img").expanduser())
FPGA = str(Path("Speck2eDevKit_1_0_1_1_0.bin").expanduser())

print_firmware_header_from_file(FX3,  FileType.fx3Firmware)
print_firmware_header_from_file(FPGA, FileType.fpgaFirmware)

d = get_programmable_devices()
if not d:
    program_empty_fx3_ram(get_empty_devices()[0], FX3)
    d = get_programmable_devices()
d = d[0]
print(d)

program_fpga_from_pc(d, FPGA)
#program_fx3_flash(d, FX3)
#program_fpga_flash(d, FPGA)
sleep(1)
d = get_programmable_devices()[0]
program_fpga_flash(d, FPGA) 

sleep(1)

d = get_programmable_devices()[0]
print_firmware_info_from_device(d)