#!/usr/bin/env python3

"""
Copyright (c) 2015, Intel Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from enum import Enum
from struct import *
import socketserver
import subprocess
import argparse
import socket
import errno
import time
import sys
import os

def truted_path_read(path):
	trusted_read_path = [
		"/sys/kernel/debug/dri",
	]

	norm_path = os.path.abspath(path)

	for trusted_path in trusted_read_path:
		if norm_path.startswith(trusted_path):
			return norm_path

	return None

def truted_path_write(path):
	trusted_write_path = [
		"/sys/devices/system/cpu/intel_pstate/",
		"/sys/kernel/mm/transparent_hugepage/enabled",
		"/sys/devices/system/cpu/cpu",
	]

	norm_path = os.path.abspath(path)

	for trusted_path in trusted_write_path:
		if norm_path.startswith(trusted_path):
			return norm_path

	return None

def is_truted_cmd(cmdline):
	if cmdline[0] == "/usr/bin/chvt":
		return len(cmdline) == 2 and cmdline[1] == '5'
	elif cmdline[0] == "/usr/bin/fgconsole":
		return len(cmdline) == 1
	elif cmdline[0] == "/usr/bin/ls":
		return True
	elif cmdline[0] == "/usr/bin/readlink":
		try:
			return len(cmdline) == 2 and os.path.abspath(cmdline[1]).startswith("/proc/")
		except:
			return False

	return False

class MessageType(Enum):
	version = 0
	file_read = 1
	file_write = 2
	cmd_exec = 3

class ErrorCode(Enum):
	OK = 0
	CMDLINE_ERROR = 1
	UNK_ERROR = 3
	NO_SERVER = 10
	VERSION_MISMATCH = 11
	OP_DENIED = 12
	ENOENT = 13
	EPERM = 14
	EPIPE = 15
	CMD_ERROR = 16

class Message:
	def __init__(self, msg_type):
		self.msg_type = msg_type

	def send(self, stream, data):
		try:
			data = data.encode()
		except AttributeError:
			pass
		msg = pack('hh', self.msg_type.value, len(data)) + data
		stream.sendall(msg)

	@classmethod
	def readBytesStream(cls, stream, count):
		buf = b''
		while count:
			newbuf = stream.recv(count)
			if not newbuf:
				raise IOError("The socket has been closed")
			buf += newbuf
			count -= len(newbuf)
		return buf

	@classmethod
	def fromStream(cls, stream):
		header = Message.readBytesStream(stream, 4)
		msg_type, payload_size = unpack("hh", header)

		payload = cls.readBytesStream(stream, payload_size)

		if msg_type == MessageType.version.value:
			return VersionRequest.fromStream(payload)
		elif msg_type == MessageType.file_read.value:
			return FileReadRequest.fromStream(payload)
		elif msg_type == MessageType.file_write.value:
			return FileWriteRequest.fromStream(payload)
		elif msg_type == MessageType.cmd_exec.value:
			return CmdExecRequest.fromStream(payload)

		return None

class VersionRequest(Message):
	def __init__(self):
		Message.__init__(self, MessageType.version)

	@classmethod
	def fromStream(cls, payload):
		return VersionRequest()

	@classmethod
	def currentVersion(cls):
		return 1

	def send(self, stream):
		super(VersionRequest, self).send(stream, "")

		try:
			return Message.readBytesStream(stream, 1)[0]
		except IOError as e:
			return ErrorCode.EPIPE, ""

	def handle_request(self, stream):
		msg = bytes({self.currentVersion()})
		stream.sendall(msg)

class FileReadRequest(Message):
	def __init__(self, path):
		Message.__init__(self, MessageType.file_read)
		self.path = path

	@classmethod
	def fromStream(cls, payload):
		return FileReadRequest(payload.decode())

	def send(self, stream):
		super(FileReadRequest, self).send(stream, self.path)

		try:
			header = Message.readBytesStream(stream, 4)
			error_code, payload_size = unpack("hh", header)
			payload = Message.readBytesStream(stream, payload_size)

			return ErrorCode(error_code), payload.decode()
		except IOError as e:
			return ErrorCode.EPIPE, ""

	def handle_request(self, stream):
		norm_path = truted_path_read(self.path)

		error_code = ErrorCode.OK.value
		data = bytes()
		if norm_path is not None:
			try:
				with open(norm_path, 'r') as f:
					data = f.read().encode()
			except OSError as err:
				error, strerror = err.args
				if error == errno.EACCES:
					error_code = ErrorCode.EPERM.value
				elif error == errno.ENOENT:
					error_code = ErrorCode.ENOENT.value
				else:
					sys.stderr.write("Unknown error caught: Errno = {}\n".format(error))
					error_code = ErrorCode.UNK_ERROR.value
			except Exception as e:
				sys.stderr.write("Unknown error caught: {}\n".format(e))
				error_code = ErrorCode.UNK_ERROR.value
		else:
			error_code = ErrorCode.OP_DENIED.value

		msg = pack('hh', error_code, len(data)) + data
		stream.sendall(msg)

class FileWriteRequest(Message):
	def __init__(self, path, data):
		Message.__init__(self, MessageType.file_write)
		self.path = path
		self.data = data

	@classmethod
	def fromStream(cls, payload):
		path_len, data_len = unpackl("hh", payload[0:4])

		path = payload[4:4+path_len]
		data = payload[4+path_len:4+path_len+data_len]

		return FileWriteRequest(path.decode(), data.decode())

	def send(self, stream):
		# Construct a write message
		path = self.path.encode()
		data = self.data.encode()
		msg = pack('hh', len(path), len(data)) + path + data
		super(FileWriteRequest, self).send(stream, msg)

		try:
			return ErrorCode(Message.readBytesStream(stream, 1)[0])
		except IOError as e:
			return ErrorCode.EPIPE, ""

	def handle_request(self, stream):
		norm_path = truted_path_write(self.path)

		error_code = ErrorCode.OK.value
		if norm_path is not None:
			try:
				with open(norm_path, 'w') as f:
					f.write(self.data)
			except OSError as err:
				error, strerror = err.args
				if error == errno.EACCES:
					error_code = ErrorCode.EPERM.value
				elif error == errno.ENOENT:
					error_code = ErrorCode.ENOENT.value
				else:
					sys.stderr.write("Unknown error caught: Errno = {}\n".format(error))
					error_code = ErrorCode.UNK_ERROR.value
			except Exception as e:
				sys.stderr.write("Unknown error caught: {}\n".format(e))
				error_code = ErrorCode.UNK_ERROR.value
		else:
			error_code = ErrorCode.OP_DENIED.value

		msg = bytes({error_code})
		stream.sendall(msg)

class CmdExecRequest(Message):
	def __init__(self, cmdline):
		Message.__init__(self, MessageType.cmd_exec)
		self.cmdline = cmdline

	@classmethod
	def fromStream(cls, payload):
		cmdline = payload.decode().split('\0')
		return CmdExecRequest(cmdline)

	def send(self, stream):
		# Construct a write message
		data = "\0".join(self.cmdline).encode()
		super(CmdExecRequest, self).send(stream, data)

		try:
			header = Message.readBytesStream(stream, 24)
			error_code, s_stderr, s_stdout = unpack("BLL", header)
			sys.stderr.write(Message.readBytesStream(stream, s_stderr).decode())
			sys.stdout.write(Message.readBytesStream(stream, s_stdout).decode())
		except IOError as e:
			print(e)
			return ErrorCode.EPIPE

		return ErrorCode(error_code)

	def handle_request(self, stream):
		error_code = ErrorCode.OK.value
		stdout = stderr = None
		if is_truted_cmd(self.cmdline):
			try:
				p = subprocess.Popen(self.cmdline, stdout=subprocess.PIPE, stderr = subprocess.PIPE)
				stdout, stderr = p.communicate()
				if p.wait() != 0:
					error_code = ErrorCode.CMD_ERROR.value
			except subprocess.CalledProcessError as e:
				error_code = ErrorCode.UNK_ERROR.value
				sys.stderr.write("Unknown process error caught: {}\n".format(e))
				pass
			except OSError as err:
				error, strerror = err.args
				if error == errno.EACCES:
					error_code = ErrorCode.EPERM.value
				elif error == errno.ENOENT:
					error_code = ErrorCode.ENOENT.value
				else:
					sys.stderr.write("Unknown OSError caught: Errno = {}\n".format(error))
					error_code = ErrorCode.UNK_ERROR.value
					pass
			except Exception as e:
				sys.stderr.write("Unknown error caught: {}\n".format(e))
				error_code = ErrorCode.UNK_ERROR.value
				pass
		else:
			error_code = ErrorCode.OP_DENIED.value

		if stderr == None:
			stderr = b""
		if stdout == None:
			stdout = b""

		msg = pack('BLL', error_code, len(stderr), len(stdout)) + stderr + stdout
		stream.sendall(msg)

class ServerHandler(socketserver.BaseRequestHandler):
	def handle(self):
		# While the socket is not closed, keep on reading messages
		try:
			while True:
				msg = Message.fromStream(self.request)
				if msg is not None:
					msg.handle_request(self.request)
		except Exception as e:
			pass

if __name__ == "__main__":
	socket_path = "\0/ezbench/rootbroker"

	# parse the options
	parser = argparse.ArgumentParser()
	parser.add_argument("-s", dest='server', help="Server mode",
						action="store_true")
	parser.add_argument("-f", dest='file', help="Set the file",
						action="store")
	parser.add_argument("-a", dest='action', help="Select the action",
						action="store", choices=('read', 'write', 'exec'))
	parser.add_argument("-d", dest='data', help="Data to write to the file",
						action="store")
	parser.add_argument("-v", dest='verbose', help="Be more verbose on error",
						action="store_true")
	parser.add_argument("cmdline", nargs='*')
	args = parser.parse_args()

	if args.server:
		server = socketserver.UnixStreamServer(socket_path, ServerHandler)
		server.serve_forever()
		sys.exit(0)

	if args.action is not None:
		# Connect to the server
		client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
		try:
			client.connect(socket_path)
		except Exception as e:
			sys.stderr.write("{}\n".format(str(e)))
			sys.exit(ErrorCode.NO_SERVER.value)

		# Verify that the server's version is compatible
		server_version = VersionRequest().send(client)
		if VersionRequest.currentVersion() != server_version:
			sys.stderr.write("ABI mismatch: Server is using {} while the client is using {}\n".format(server_version, VersionRequest.currentVersion()))
			sys.exit(ErrorCode.VERSION_MISMATCH.value)

		msg = None
		if args.action == "read":
			if args.file is None:
				sys.stderr.write("Error: Action 'read' requires a file (-f)\n")
				sys.exit(ErrorCode.CMDLINE_ERROR)
			msg = FileReadRequest(args.file)
			error, data = msg.send(client)
			if error == ErrorCode.OK:
				sys.stdout.write(data)
			else:
				sys.stderr.write("Error: {}\n".format(str(error)))

			sys.exit(error.value)
		elif args.action == "write":
			if args.file is None:
				sys.stderr.write("Error: Action 'write' requires a file (-f)\n")
				sys.exit(ErrorCode.CMDLINE_ERROR)
			msg = FileWriteRequest(args.file, args.data)
			error = msg.send(client)
			if error != ErrorCode.OK:
				sys.stderr.write("Error: {}\n".format(str(error)))

			sys.exit(error.value)
		elif args.action == "exec":
			msg = CmdExecRequest(args.cmdline)
			error = msg.send(client)
			if error != ErrorCode.OK and args.verbose:
				sys.stderr.write("Error: {}\n".format(str(error)))

			sys.exit(error.value)
