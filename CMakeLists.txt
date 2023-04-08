CMAKE_MINIMUM_REQUIRED (VERSION 3.0)

PROJECT (DarknetShowcase C CXX)

SET (CMAKE_BUILD_TYPE Release)
SET (CMAKE_CXX_STANDARD 17)
SET (CMAKE_CXX_STANDARD_REQUIRED ON)

ADD_DEFINITIONS ("-Wall -Wextra -Werror -Wno-unused-parameter")

FIND_PACKAGE (Threads		REQUIRED	)
FIND_PACKAGE (OpenCV		REQUIRED	)
FIND_LIBRARY (MOVEDETECT	movedetect	) # https://github.com/stephanecharette/MoveDetect#how-to-build-movedetect
FIND_LIBRARY (DARKHELP		darkhelp	) # https://github.com/stephanecharette/DarkHelp#building-darkhelp-linux
FIND_LIBRARY (DARKNET		darknet		) # https://github.com/stephanecharette/DarkHelp#building-darknet-linux

INCLUDE_DIRECTORIES (${OpenCV_INCLUDE_DIRS})

FILE (GLOB SOURCE *.cpp)
LIST (SORT SOURCE)

ADD_EXECUTABLE (dnshowcase ${SOURCE})
TARGET_LINK_LIBRARIES (dnshowcase Threads::Threads ${OpenCV_LIBS} ${MOVEDETECT} ${DARKHELP} ${DARKNET})
