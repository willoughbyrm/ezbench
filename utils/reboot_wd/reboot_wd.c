#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <linux/watchdog.h>
#include <sys/ioctl.h>

int usage(const char* exe)
{
  printf("Usage: %s timeout\n", exe);
  printf(" Opens all hardware watchdogs that can be found and sets them to timeout in `timeout' seconds.\n");
  printf(" There's no way out; This command is meant for ensuring a reboot really happens.\n");
  return 1;
}

int main(int argc, char** argv)
{
  int fd;
  int timeout;
  int i;
  char buf[255];

  if (argc < 2) {
    exit(usage(argv[0]));
  }

  timeout = atoi(argv[1]);
  if (timeout <= 0) {
    fprintf(stderr, "Error: timeout must be positive and non-zero\n");
    exit(usage(argv[0]));
  }

  for (i = 0; i < 25; ++i) {
    snprintf(buf, 255, "/dev/watchdog%d", i);
    fd = open(buf, O_WRONLY);
    if (fd >= 0) {
      printf("reboot_wd: Setting timeout on watchdog device %s\n", buf);
      ioctl(fd, WDIOC_SETTIMEOUT, &timeout);
      close(fd);
    }
  }

  return 0;
}
