#!/usr/bin/env python3
import os
import subprocess
import sys
import re
import json
import time

def find_visual_studio():
    """Find Visual Studio installation and initialize environment."""
    print("Searching for Visual Studio...")

    vs_paths = [
        r"C:\Program Files\Microsoft Visual Studio\{}\Community\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\{}\Community\VC\Auxiliary\Build\vcvars64.bat"
    ]

    for year in ["2022", "2019", "2017"]:
        for path_template in vs_paths:
            vs_path = path_template.format(year)
            if os.path.exists(vs_path):
                print(f"Found Visual Studio {year}")
                return vs_path

    print("Error: Could not find Visual Studio environment.")
    sys.exit(1)

def get_vs_environment(vs_path):
    """Get Visual Studio environment variables, with caching."""
    cache_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".vsEnvironmentCache.json")
    cache_max_age = 24 * 60 * 60  # 24 hours in seconds

    # Check if we have a valid cache
    if os.path.exists(cache_file):
        try:
            cache_stat = os.stat(cache_file)
            if time.time() - cache_stat.st_mtime < cache_max_age:
                with open(cache_file, 'r') as f:
                    print("Using cached VS environment")
                    return json.load(f)
        except (OSError, json.JSONDecodeError):
            # If any error occurs, just regenerate the cache
            pass

    print("Generating VS environment...")
    # Get the VS environment variables
    cmd = f'"{vs_path}" && set'
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    stdout, _ = proc.communicate()

    # Parse environment variables
    env = {}
    for line in stdout.decode().splitlines():
        if '=' in line:
            key, value = line.split('=', 1)
            env[key] = value

    # Cache the environment
    try:
        with open(cache_file, 'w') as f:
            json.dump(env, f)
    except OSError:
        print("Warning: Could not write VS environment cache")

    return env

def execute_command():
    """Execute the provided command with Visual Studio environment."""
    vs_path = find_visual_studio()

    # Add CMake to PATH
    cmake_path = r"C:\Program Files\CMake\bin"
    if os.path.exists(cmake_path):
        os.environ["PATH"] = os.environ["PATH"] + os.pathsep + cmake_path

    # If no arguments provided, exit
    if len(sys.argv) <= 1:
        print("No command provided")
        return 0

    # Get VS environment variables with caching
    vs_env = get_vs_environment(vs_path)

    # Create final environment by merging current with VS environment
    env = os.environ.copy()
    env.update(vs_env)

    # Execute the user command with VS environment
    cmd = ' '.join(sys.argv[1:])
    print(f"Running command: {cmd}")
    return subprocess.call(cmd, env=env, shell=True)

if __name__ == "__main__":
    sys.exit(execute_command())