import os
import numpy as np


class EventHandler(object):

    def __init__(self):
        super(EventHandler, self).__setattr__('_events', {})

    def __getattr__(self, name):
        def deco(f):
            self._events[name] = f
            return f
        return deco

    def __setattr__(self, name, f):
        self._events[name] = f

    def on(self, name, *args, **kwargs):
        event = self._events.get(name, None)
        if event:
            event(*args, **kwargs)
