# -*- coding: utf-8 -*-
import json
import logging

from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from converter.core.ascii2segy import get_frame, checks, write_segy_file


logger = logging.getLogger(__name__)


def index(request):
    return render(request, 'index.html')


@csrf_exempt
def convert_file(request):
    try:
        array = json.loads(request.body)
    except ValueError:
        return HttpResponse(status=400)

    try:
        array = [list(map(lambda x: float(x), y)) for y in array]
        data = get_frame(array)
        checks(data)
    except Exception as e:
        logger.exception(e)
        return HttpResponse(e.args[0], status=400)

    try:
        write_segy_file(data, 'test.segy')
    except Exception as e:
        logger.exception(e)
        return HttpResponse(e.args[0], status=500)
    return HttpResponse(status=200)
