/* Copyright (c) 2017, Intel Corporation
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright notice,
 *       this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of Intel Corporation nor the names of its contributors
 *       may be used to endorse or promote products derived from this software
 *       without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <fcntl.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <linux/watchdog.h>
#include <sys/ioctl.h>
#include <sys/select.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <stdarg.h>

int usage(const char* exe)
{
  printf("Usage: %s timeout command [parameters]\n", exe);
  printf(" Executes command and watches for output.\n");
  printf(" timeout - maximum time to wait for output before the process is killed.\n");
  return 1;
}

void log_msg(const char *fmt, ...) {
  va_list arg;

  /* Write the error message */
  va_start(arg, fmt);
  fprintf(stdout, "owatch: ");
  vfprintf(stdout, fmt, arg);
  va_end(arg);
  fflush(stdout);


  FILE *f = fopen("/dev/kmsg", "w");
  if (f) {
    /* Write the message to the kernel logs */
    fprintf(f, "owatch: ");
    va_start(arg, fmt);
    vfprintf(f, fmt, arg);
    va_end(arg);

    fflush(f);
    fsync(fileno(f));
    fclose(f);
  }
}

#define NUM_WDS 25
int _watchdogfd[NUM_WDS];
int _wdissoft[NUM_WDS];

void wd_close_atexit();
void wd_close(int);

void init_wd()
{
  int i;

  for (i = 0; i < NUM_WDS; ++i) {
    _watchdogfd[i] = -1;
    _wdissoft[i] = 0;
  }

  /* make sure the watchdog are closed when exiting */
  atexit(wd_close_atexit);
  signal(SIGTERM, wd_close);
  signal(SIGINT, wd_close);
  signal(SIGQUIT, wd_close);
}

int wd_settimeout(int timeout)
{
  int ret, i, valids = 0, origtimeout;

  for (i = 0; i < NUM_WDS; ++i) {
    if (_watchdogfd[i] >= 0) {
      if (!_wdissoft[i]) {
	/* Increase timeout for watchdogs that are not softdogs so
	 * softdog triggers first. A softdog trigger will cause a
	 * panic, and we want to see that in the logs.
	 */

	/* TODO: Only do the increase if there is also a softdog.
	 */
        timeout += 10; /* TODO: Magic number */
      }
      origtimeout = timeout;
      ret = ioctl(_watchdogfd[i], WDIOC_SETTIMEOUT, &timeout);
      if (!ret) {
        ++valids;
        log_msg("timeout for /dev/watchdog%d set to %d (requested %d)\n", i, timeout, origtimeout);
      }
      else {
          log_msg("timeout %d not accepted by /dev/watchdog%d, closing gracefully\n",
              timeout, i);
          write(_watchdogfd[i], "V", 1);
          close(_watchdogfd[i]);
          _watchdogfd[i] = -1;
      }
    }
  }

  return valids;
}

void wd_heartbeat()
{
  int ret, i;

  for (i = 0; i < NUM_WDS; ++i) {
    if (_watchdogfd[i] >= 0) {
      ret = ioctl(_watchdogfd[i], WDIOC_KEEPALIVE, 0);
      if (ret) {
        /* WARNING:do not close the watchdog as we are never supposed to fail and
         * we rather would like to die, than potentially getting stuck later on.
         */
        log_msg("/dev/watchdog%d heartbeat failed (ret=%i)\n", i, ret);
      }
    }
  }
}

void wd_close(int signal)
{
  int i;

  for (i = 0; i < NUM_WDS; ++i) {
    if (_watchdogfd[i] < 0)
      continue;

    write(_watchdogfd[i], "V", 1);
    close(_watchdogfd[i]);
    _watchdogfd[i] = -1;
    if (!signal)
      log_msg("/dev/watchdog%d closed\n", i);
  }
}

void wd_close_atexit()
{
  wd_close(0);
}

void open_watchdog_dev(int timeout)
{
  int fd;
  char buf[255];
  int i, ret, valids, opened = 0;
  struct watchdog_info info;

  for (i = 0; i < NUM_WDS; ++i) {
    snprintf(buf, 255, "/dev/watchdog%d", i);
    fd = open(buf, O_WRONLY);
    if (fd >= 0) {
      log_msg("Using watchdog device %s\n", buf);
      _watchdogfd[i] = fd;
      ++opened;

      ret = ioctl(fd, WDIOC_GETSUPPORT, &info);
      if (!ret && !strcmp((char*)info.identity, "Software Watchdog")) {
        _wdissoft[i] = 1;
        log_msg("Watchdog /dev/watchdog%d is a software watchdog\n", i);
      }
    }
  }

  valids = wd_settimeout(timeout);
  if (valids != opened) {
    log_msg("Only %d/%d watchdogs kept\n", valids, opened);
  }
  wd_heartbeat();
}

void sysrq_write(const char* cmd)
{
  FILE* sysrq = fopen("/proc/sysrq-trigger", "w");
  if (sysrq) {
    fprintf(sysrq, "%s", cmd);
    fclose(sysrq);
  }
}

void sync_and_panic()
{
  sysrq_write("c");
  sysrq_write("s");
  sysrq_write("u");
  sysrq_write("b");
}

bool have_any_wd()
{
  int i;

  for (i = 0; i < NUM_WDS; ++i) {
    if (_watchdogfd[i] >= 0)
      return true;
  }

  return false;
}

void watchdog_die(int timeout)
{
  if (have_any_wd()) {
    wd_settimeout(1);
    sleep(10);
  }
}

/*
 * return 0 if no output occurred, 1 if it did.
 * -1 on eof on either fd, -2 for other errors
 */
int pipe_output(int timeout, int out, int err)
{
  struct timeval tv = { .tv_sec = timeout };
  fd_set set;
  int nfds = out > err ? out + 1 : err + 1;
  int n, ret;
  char buf[512];

  FD_ZERO(&set);
  FD_SET(out, &set);
  FD_SET(err, &set);

  n = select(nfds, &set, NULL, NULL, &tv);
  if (n < 0) {
    log_msg("select() failed in pipe_output(): %s\n", strerror(n));
    return -2;
  }
  if (!n) {
    return 0;
  }

  ret = -1;
  if (FD_ISSET(out, &set)) {
    ssize_t s = read(out, buf, sizeof(buf));
    if (s < 0) {
      log_msg("reading stdout failed in pipe_output(): %s\n", strerror(n));
      return -2;
    }

    if (s > 0) {
      write(STDOUT_FILENO, buf, s);
      ret = 1;
    }
  }
  if (FD_ISSET(err, &set)) {
    ssize_t s = read(err, buf, sizeof(buf));
    if (s < 0) {
      log_msg("reading stderr failed in pipe_output(): %s\n", strerror(n));
      perror("read");
      return -2;
    }

    if (s > 0) {
      write(STDERR_FILENO, buf, s);
      ret = 1;
    }
  }

  return ret;
}

void overwatch(pid_t child, int timeout, int outpipe[2], int errpipe[2])
{
  int n = 1;
  int wstatus;
  pid_t r;
  pid_t killtarget = child;

  close(outpipe[1]);
  close(errpipe[1]);

  /* Set the timeout to $timeout + 10s to give ourselves some time to close
   * things properly if the machine allows us
   */
  open_watchdog_dev(timeout + 10);

  while (n > 0) {
    wd_heartbeat();
    n = pipe_output(timeout, outpipe[0], errpipe[0]);
  }

  close(outpipe[0]);
  close(errpipe[0]);

  if (n == 0) {
    log_msg("TIMEOUT!\n");

    /* One more heartbeat to keep the machine alive for debug info */
    wd_settimeout(10);
    wd_heartbeat();

    /* TODO: Force a sync, then a panic to make sure logs are being saved to pstore */
    sync_and_panic();

    /* We only reach this point if the above function could not boot the machine. Example reasons:
     * No /proc/sysrq-trigger, not running as root
     */

    /* Hack: If we have a hw watchdog, don't bother killing children.
     * Just stop the heartbeat. */
    watchdog_die(10);

    /* We only reach this point if 1) sysrq above did not boot 2) we didn't
     * have watchdogs that could boot the machine. */

    log_msg("Killing children\n");

    if (!kill(-child, 0)) {
      /* Child was able to setsid, process group exists.
       * Use process group as kill target.
       */
      killtarget = -child;
    }

    kill(killtarget, 15);
  }

  r = waitpid(child, &wstatus, WNOHANG);

  if (r == 0) {
    wd_settimeout(30);
    wd_heartbeat();
    kill(killtarget, 9);
    r = waitpid(child, &wstatus, 0);
    wd_heartbeat();
  }

  wd_close(0);

  if (r != child) {
    log_msg("Child turned undead, hire a priest\n");
    exit(1);
  }

  if (n == -1) {
    /* normal termination */
    if (WIFEXITED(wstatus)) {
      exit(WEXITSTATUS(wstatus));
    } else {
      exit(1);
    }
  } else if (n == -2) {
    /* error occurred */
    exit(1);
  }

  /* shouldn't be reached */
  exit(7);
}

void launch_child(int outpipe[2], int errpipe[2], char** argv)
{
  pid_t sid;

  close(outpipe[0]);
  close(errpipe[0]);

  sid = setsid();
  if (sid < 0) {
    perror("setsid");
    /* continue anyway */
  }

  if (dup2(outpipe[1], STDOUT_FILENO) < 0 ||
      dup2(errpipe[1], STDERR_FILENO) < 0) {
    perror("dup2");
    exit(1);
  }

  close(outpipe[1]);
  close(errpipe[1]);

  execvp(argv[0], argv);

  perror("execvp");
  exit(1);
}

int main(int argc, char** argv)
{
  int outpipe[2];
  int errpipe[2];
  int timeout;
  pid_t child;

  init_wd();

  if (argc < 3) {
    exit(usage(argv[0]));
  }

  timeout = atoi(argv[1]);
  if (timeout <= 0) {
    fprintf(stderr, "Error: timeout must be positive and non-zero\n");
    exit(1);
  }

  if (pipe(outpipe) || pipe(errpipe)) {
    perror("pipe");
    exit(1);
  }

  if ((child = fork())) {
    if (child < 0) {
      perror("fork");
      exit(1);
    }

    overwatch(child, timeout, outpipe, errpipe);
  } else {
    launch_child(outpipe, errpipe, &argv[2]);
  }

  __builtin_unreachable();
  return 255;
}
