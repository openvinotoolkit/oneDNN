@ECHO off
SETLOCAL

::===============================================================================
:: Copyright 2019-2023 Intel Corporation
::
:: Licensed under the Apache License, Version 2.0 (the "License");
:: you may not use this file except in compliance with the License.
:: You may obtain a copy of the License at
::
::     http://www.apache.org/licenses/LICENSE-2.0
::
:: Unless required by applicable law or agreed to in writing, software
:: distributed under the License is distributed on an "AS IS" BASIS,
:: WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
:: See the License for the specific language governing permissions and
:: limitations under the License.
::===============================================================================

:process_arguments
IF "%1" == "/BUILDDIR" SET "BUILDDIR=%2"
IF "%1" == "/MODE" SET "MODE=%2"
IF "%1" == "/REPORTDIR" SET "REPORTDIR=%2"

SHIFT
SHIFT
IF NOT "%1" == "" GOTO process_arguments

SET "CTEST_OPTS=--output-on-failure"

IF NOT "%REPORTDIR%" == "" SET "GTEST_OUTPUT=%REPORTDIR%\report\test_report.xml"

IF NOT "%LLVM_PATH%" == "" SET "PATH=%LLVM_PATH%\bin;%PATH%"
SET "PATH=%BUILDDIR%\src;%BUILDDIR%\src\%MODE%;%PATH%"
SET "LIB=%BUILDDIR%\src;%BUILDDIR%\src\%MODE%;%LIB%"

ECHO "CTEST OPTIONS: %CTEST_OPTS%"

CD /D %BUILDDIR%

IF "%ONEDNN_SKIP_TESTS_REGEX%" == "" (
    ctest %CTEST_OPTS%
) ELSE (
    ctest %CTEST_OPTS% -E "%ONEDNN_SKIP_TESTS_REGEX%"
)

echo "DONE"
