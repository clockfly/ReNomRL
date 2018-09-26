import os
import numpy as np


class EventHandler(object):

    def __init__(self, events):
        super(EventHandler, self).__setattr__('_events', events)

    def __getattr__(self, name):
        def deco(f):
            self._events[name] = f
            return f
        return deco

    def __setattr__(self, name, f):
        self._events[name] = f

    def get_handlers(self):
        return self._evnets

    def on(self, name, *args, **kwargs):
        event = self._evnets.get(name)
        if event:
            event(*args, **kwargs)
