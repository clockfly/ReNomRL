import renom as rm

class ActorCriticModelA3C(rm.Model):
    """
    This class creates actor critic model easily. If model2 is defined, then they are used seperately, but with common input.
    However, if model2 is not defined, model and model2 will share the same structure.

    still under construction.


    Args:
        model: actor model.
        model2: critic model.(optional)

    Examples:
        >>> from renom_rl.utility.structure import ActorCriticModelA3C
        >>> import renom as rm
        >>>
        >>> model_common=rm.Sequential([
        ...     rm.Dense(32),
        ...     rm.Relu(),
        ...     rm.Dense(32),
        ...     rm.Relu(),
        ...     rm.Dense(3),
        ... ])
        >>>
        >>> model_ac=ActorCriticModelA3C(model_common)


    """

    def __init__(self, model, model2=None):

        self.model = model
        self.model2 = model2

        self.forward_func = self.forward_common

        if model2 is not None:
            if isinstance(model, rm.Sequential):
                assert model[-1]._output_size > 1, "model argument needs more that 1 output"

            if isinstance(model2, rm.Sequential):
                assert model2[-1]._output_size == 1, "model2 argument needs only 1 output"

            self.func = self.forward_sep

            # for i,ly in enumerate(model2.iter_models()):
            #     setattr(self, "mem2_%d" % (i), ly)

        else:
            if isinstance(model, rm.Sequential):
                assert model[-1]._output_size > 2, "model argument needs more that 2 output"


        # for i,ly in enumerate(model.iter_models()):
        #     setattr(self, "mem1_%d" % (i), ly)


    def forward(self, X):

        return self.forward_func(X)

    def forward_common(self, X):
        y = self.model(X)
        act = rm.softmax(y[:, 0:-1])
        val = y[:, -1].reshape(-1,1)
        return act, val

    def forward_sep(self, X):
        y1 = rm.softmax(self.model(X))
        y2 = self.model2(X)
        return y1, y2

    def forward(self, X):
        y = self.model(X)
        act = rm.softmax(y[:, 0:-1])
        val = y[:, -1].reshape(-1,1)
        return act, val


# class ActorCriticModelA3C_2(rm.Model):
#     """
#     This class creates actor critic model easily. If model2 is defined, then they are used seperately, but with common input.
#     However, if model2 is not defined, model and model2 will share the same structure.
#
#     still under construction.
#
#
#     Args:
#         model: actor model.
#         model2: critic model.(optional)
#
#     Examples:
#         >>> from renom_rl.utility.structure import ActorCriticModelA3C
#         >>> import renom as rm
#         >>>
#         >>> model_common=rm.Sequential([
#         ...     rm.Dense(32),
#         ...     rm.Relu(),
#         ...     rm.Dense(32),
#         ...     rm.Relu(),
#         ...
#         ... ])
#         >>>
#         >>> model_ac=ActorCriticModelA3C(model_common)
#
#
#     """
#
#     def __init__(self, model, actor, model2=None):
#
#         self.model = model
#         # self.model2 = model2
#         self.layer_actor = rm.Dense(actor)
#         self.layer_value = rm.Dense(1)
#
#
#     def forward(self, X):
#         y = self.model(X)
#         act = rm.softmax(self.layer_actor(y))
#         val = self.layer_value(y)
#         return act, val
