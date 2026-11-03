#!/usr/bin/env bash
# Builds the command line tool (bin/sbdv) and the shared library (bin/libsbdv.so).  Needs only a C++17 compiler (g++ or clang++).
#   ./build.sh            portable build
#   NATIVE=1 ./build.sh   tuned for this machine's CPU (a little faster, not portable to other CPUs)
#   DEBUG_RING=1 ./build.sh   adds a check that every frame read is inside the ring buffer (used by the tests, slower)
set -e
cd "$(dirname "$0")"; mkdir -p bin
CXX=${CXX:-g++}; FLAGS="-O3 -std=c++17"; [ -n "$NATIVE" ] && FLAGS="$FLAGS -march=native"; [ -n "$DEBUG_RING" ] && FLAGS="$FLAGS -DSBDV_DEBUG_RING"
OUT=${OUT:-bin}; mkdir -p "$OUT"
$CXX $FLAGS -Iinclude -Isrc src/sbdv_cli.cpp -o "$OUT/sbdv"
case "$(uname)" in Darwin) LIB=libsbdv.dylib;; *) LIB=libsbdv.so;; esac
$CXX $FLAGS -shared -fPIC -fvisibility=hidden -Iinclude -Isrc src/sbdv_capi.cpp -o "$OUT/$LIB"
echo "built: $OUT/sbdv  $OUT/$LIB  ($("$OUT/sbdv" --version))"
