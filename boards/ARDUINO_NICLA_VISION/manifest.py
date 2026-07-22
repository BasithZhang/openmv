# Minimal frozen modules for the Nicla YOLO-Pro firmware.
# _boot.py mounts the filesystem; machine.py provides the LED helper.
freeze("$(OMV_LIB_DIR)/", "_boot.py")
freeze("$(OMV_LIB_DIR)/", "machine.py")
