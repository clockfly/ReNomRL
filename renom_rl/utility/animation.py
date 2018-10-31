import matplotlib.pyplot as plt
from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from IPython.display import display


class Animation(object):
    """Animation Class.

    This class has objects that can save images from open AI gym, etc.
    Because render can only play the render peroformance once, we prepared
    this module, so that we can reanimate the performance without
    re-testing.

    Args:
        dpi(int): dots per inch. resolution.
        ratio(int): reduces scale.
        interval(int): play rate[ms]


    Example:
        >>> from renom_rl.utility.animation import Animation
        >>> import random
        >>> import gym
        >>>
        >>> env=gym.make("CartPole-v0")
        >>> env.reset()
        >>>
        >>> animation=Animation()
        >>>
        >>> for i in range(200):
        ...     action = random.choice([0,1])
        ...     step,_ = env.step(action)
        ...     image = env.render(mode="rgb_array")
        ...     animation.store(image)
        >>>
        >>> env.close()
        >>> animation.run(reset=True)
        ...
    """

    def __init__(self, dpi=72.0, ratio=72.0, interval=50):
        self.frames = []
        self.dpi = dpi
        self.ratio = ratio
        self.interval = interval

    def __len__(self):
        return len(self.frames)

    def store(self, frame):
        """
        This function stores the image (render) data to a lists.
        """
        self.frames.append(frame)

    def run(self, reset=False):
        """
        This function creates the animation. The animation frame size is based on the first element of the stored image.
        Users can use the reset option in order to reset the stored image. Note that only the stored image will reset
        but the animation will be kept. If users redo this function with ``reset = True``, users will get an error.

        Args:
            reset(boolean): resets at the end of animation.

        """
        assert len(self) > 0, "No length"
        frame_r = self.frames

        plt.figure(figsize=(frame_r[0].shape[1] / self.ratio,
                            frame_r[0].shape[0] / self.ratio), dpi=self.dpi)
        image = plt.imshow(frame_r[0])
        plt.axis("off")

        def animate(i):
            image.set_data(frame_r[i])

        anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frame_r), interval=50)

        # #saves animation
        # if save==True:
        #     anim.save(name+".mp4")

        # anim.save("movie_cartpole.mp4")
        display(display_animation(anim))

        if reset:
            self.reset()

    def reset(self):
        """
        This function resets the list of stored image.
        """
        self.frames = []
