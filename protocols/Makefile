PROTOS     := $(wildcard *.proto)
PROTOS_OUT := $(PROTOS:.proto=_pb2.py)

%_pb2.py: %.proto
	protoc -I=. --python_out=. $<

all: $(PROTOS_OUT)

clean:
	rm $(PROTOS_OUT)
