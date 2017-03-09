
all: reboot_wd

reboot_wd.o: reboot_wd.c
	$(CC) -c -Wall -o $@ $<

reboot_wd: reboot_wd.o
	$(CC) -o $@ $<
