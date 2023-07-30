"""This here is to test the env. You can also just run the env,
but this includes random actions and making a video."""

import datetime

import numpy as np
import cv2

from tripod_env import env


VIDEO_NAME = f'{datetime.datetime.now()}.mp4'
LENGTH = 500
WIDTH = 600
HEIGHT = 480


def convert_rgb(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def render():
    video_writer = cv2.VideoWriter(VIDEO_NAME,
        cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (WIDTH, HEIGHT))

    action_spec = env.action_spec()
    time_step = env.reset()
    for _ in range(LENGTH):
        # Random action
        action = np.random.uniform(action_spec.minimum,
                                   action_spec.maximum,
                                   size=action_spec.shape)

        # Test if legs are strong enough to push up the base
        n_legs = action_spec.shape[0] // 4
        action = (0, 0.7, 0, 0.7) * n_legs
        # print(action)

        # Step
        time_step = env.step(action)
        obs = [value[0] for value in time_step.observation.values()]
        # flat_obs = [el1 for el in obs for el1 in el]
        reward = time_step.reward

        frame = env.physics.render(HEIGHT, WIDTH)
        video_writer.write(convert_rgb(frame))
        # video_writer.write(grab_frame(env))

    # End render to video file
    video_writer.release()


def play():
    # Second pass - Playback
    cap = cv2.VideoCapture(VIDEO_NAME)
    while(cap.isOpened()):
        ret, frame = cap.read()
        cv2.imshow('Playback', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

    # Exit
    cv2.destroyAllWindows()


render()
print('video made')
# play()
