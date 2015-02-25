COMPILER = clang++
CFLAGS   = -std=c++11 -ftemplate-depth=512 -Wall -MMD -MP -O3 #-fopenmp#-g
LDFLAGS  = $(shell pkg-config --libs opencv) -L/usr/local/lib -lboost_system -lboost_filesystem -lboost_graph -lboost_serialization ./blas/libblas.a
LIBS     = 
INCLUDE  = -Iliblinear/include -Iblas -Iinclude $(shell pkg-config --cflags opencv eigen3)
TARGET   = libcv_wrapper.a
TARGETTEST = test
OBJDIR   = ./obj
ifeq "$(strip $(OBJDIR))" ""
	OBJDIR = .
endif
SOURCES  = $(wildcard src/cv_wrapper/*.cpp liblinear/src/*.cpp) sample.cpp
OBJECTS  = $(addprefix $(OBJDIR)/, $(SOURCES:.cpp=.o))
DEPENDS  = $(OBJECTS:.o=.d)

$(TARGETTEST): $(OBJECTS) $(LIBS)
	$(COMPILER) -o $@ $^ $(LDFLAGS)

$(TARGET): $(OBJECTS) $(LIBS)
	@[ -d $(dir $@) ] || mkdir -p $(dir $@)
	ar r $@ $^
#	$(COMPILER) -o $@ $^ $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp
	@[ -d $(dir $@) ] || mkdir -p $(dir $@)
	$(COMPILER) $(CFLAGS) $(INCLUDE) -o $@ -c $<

all : $(TARGET) $(TARGETTEST)

clean:
	rm -f $(OBJECTS) $(DEPENDS) $(TARGET)
	@rmdir --ignore-fail-on-non-empty $(OBJDIR)

-include $(DEPENDS)
