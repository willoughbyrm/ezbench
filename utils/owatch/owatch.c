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

int usage(const char* exe)
{
  printf("Usage: %s timeout command [parameters]\n", exe);
  printf(" Executes command and watches for output.\n");
  printf(" timeout - maximum time to wait for output before the process is killed.\n");
  return 1;
}

#define NUM_WDS 25
int _watchdogfd[NUM_WDS];

void init_wd()
{
  int i;

  for (i = 0; i < NUM_WDS; ++i) {
    _watchdogfd[i] = -1;
  }
}

void wd_settimeout(int timeout)
{
  int i;

  for (i = 0; i < NUM_WDS; ++i) {
    if (_watchdogfd[i] >= 0)
      ioctl(_watchdogfd[i], WDIOC_SETTIMEOUT, &timeout);
  }
}

void wd_heartbeat()
{
  int i;

  for (i = 0; i < NUM_WDS; ++i) {
    if (_watchdogfd[i] >= 0)
      ioctl(_watchdogfd[i], WDIOC_KEEPALIVE, 0);
  }
}

void wd_close()
{
  int i;

  for (i = 0; i < NUM_WDS; ++i) {
    if (_watchdogfd[i] < 0)
      continue;

    write(_watchdogfd[i], "V", 1);
    close(_watchdogfd[i]);
    _watchdogfd[i] = -1;
  }
}

void open_watchdog_dev(int timeout)
{
  int fd;
  char buf[255];
  int i;

  for (i = 0; i < NUM_WDS; ++i) {
    snprintf(buf, 255, "/dev/watchdog%d", i);
    fd = open(buf, O_WRONLY);
    if (fd >= 0) {
      printf("owatch: Using watchdog device %s\n", buf);
      _watchdogfd[i] = fd;
    }
  }

  wd_settimeout(timeout);
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
    sleep(3);
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
    perror("select");
    return -2;
  }
  if (!n) {
    return 0;
  }

  ret = -1;
  if (FD_ISSET(out, &set)) {
    ssize_t s = read(out, buf, sizeof(buf));
    if (s < 0) {
      perror("read");
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

  open_watchdog_dev(timeout);

  while (n > 0) {
    wd_heartbeat();
    n = pipe_output(timeout, outpipe[0], errpipe[0]);
  }

  wd_heartbeat();

  close(outpipe[0]);
  close(errpipe[0]);

  if (n == 0) {
    printf("owatch: TIMEOUT!\n");

    /* Hack: If we have a hw watchdog, don't bother killing children.
     * Just stop the heartbeat. */
    watchdog_die(3);

    printf("owatch: Killing children\n");

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

  wd_close();

  if (r != child) {
    printf("Child turned undead, hire a priest\n");
    exit(1);
  }

  if (n == -1) {
    /* normal termination */
    if (WIFEXITED(wstatus)) {
      exit(WEXITSTATUS(wstatus));
    } else {
      exit(1);
    }
  }

  if (n == -2) {
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
