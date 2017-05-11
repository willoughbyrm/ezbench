#!/usr/bin/env python3

import os
import subprocess

def __run_command__(cmd):
    try:
        pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        stdout, stderr = pipe.communicate()
        stdout = stdout.decode()
        stderr = stderr.decode()

        return pipe.returncode, stdout, stderr
    except Exception as e:
        print("compare_image: Failed to run the command '{}': {}".format(cmd, e))
        return -1, "", ""

def compare_image(image1, image2, metric, output):
    """
    Compare two images with given metric

    Args:
        image1: First image file to compare
        image2: Second image file to compare
        metric: List of metrics to pass to ImageMagick compare
        output: Optional output file, can be 'null:'

    Returns:
        Difference between images in given metric
    """

    args = ['compare', '-metric']
    args.extend(metric)
    args.extend([image1, image2, output])

    returncode, stdout, stderr = __run_command__(args)
    if returncode != 0 and returncode != 1:
        return -1

    # Remove parentheses from the output of compare: "value (normalized value)"
    diffs = stderr.replace('(', '').replace(')', '').split(' ')
    if len(diffs) == 0:
        return -1

    try:
        return float(diffs[-1])
    except ValueError:
        return -1

def average_image(images, output):
    """
    Create an average image out of an image list

    Args:
        images: List of images to average
        output: Output filename

    Returns:
        True on success, False on failure
    """

    args = ['convert', '-average']
    args.extend(images)
    args.append(output)

    returncode, stdout, stderr = __run_command__(args)
    return returncode == 0

def compare(image1, image2, metric, output, reset_cache=False):
    """
    Compare two images and cache the result

    Args:
        image1: First image file to compare
        image2: Second image file to compare
        metric: List of metrics to pass to ImageMagick compare
        output: Optional output file, can be 'null:'

    Returns:
        Difference between images in given metric
    """

    diff = 0.0

    cache = '{}_compare_{}'.format(os.path.splitext(image1)[0],
            os.path.basename(os.path.splitext(image2)[0]))
    if not reset_cache and os.path.exists(cache):
        try:
            return float(open(cache, 'rt').read())
        except:
            print("compare: Invalid cached value in file '{}'".format(cache))

    diff = compare_image(image1, image2, metric, output)
    try:
        open(cache, 'wt').write(str(diff))
    except Exception as e:
        print("compare: Failed to write to the cache file '{}': {}".format(cache, e))

    return diff

def average(images, output, reset_cache=False):
    """
    Create an average image out of an image list and cache the result

    Args:
        images: List of images to average
        output: Output filename

    Returns:
        True on success, False on failure
    """

    # Sort images so the order is correct when comparing against cache
    images = list(images)
    images.sort()

    cache = os.path.splitext(output)[0]
    if not reset_cache and os.path.exists(cache):
        files = open(cache, 'rt').read()
        if str(images) == files:
            return

    average_image(images, output)

    try:
        open(cache, 'wt').write(str(images))
    except Exception as e:
            print("compare: Failed to write to the cache file '{}': {}".format(cache, e))
