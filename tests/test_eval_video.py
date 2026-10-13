from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

import imageio_ffmpeg
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from eval_video import EpisodeVideo
from excavator_common.bridge_protocol import encode_rgb_payload


class EpisodeVideoTests(unittest.TestCase):
    def make_video(self, path):
        return EpisodeVideo(path, episode=1, seed=2, fps=5, max_steps=350, min_particles=100)

    def reply(self):
        colors = {'0': (0, 255, 0), '1': (0, 0, 255), '2': (255, 255, 0)}
        return {
            'evaluation_video_frame': encode_rgb_payload(np.full((16, 16, 3), (255, 0, 0), dtype=np.uint8)),
            'cameras': {name: encode_rgb_payload(np.full((16, 16, 3), color, dtype=np.uint8))
                        for name, color in colors.items()},
        }

    def read_video(self, path):
        probe = subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), '-i', str(path), '-f', 'null', '-'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True,
        ).stderr
        size = re.search(r', (\d+)x(\d+)[ ,]', probe)
        fps = re.search(r'(\d+(?:\.\d+)?) fps', probe)
        frames = re.findall(r'frame=\s*(\d+)', probe)
        return {'size': tuple(map(int, size.groups())), 'fps': float(fps.group(1))}, int(frames[-1])

    def test_camera_panels_and_success_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = self.make_video(Path(tmp) / 'evaluation.mp4')
            status = {'status': 'running', 'policy_steps_executed': 1}
            video.write(self.reply(), status)
            frame = video.render(status)
            for y, x, color in [(400, 400, (255, 0, 0)), (128, 896, (0, 255, 0)),
                                (384, 896, (0, 0, 255)), (640, 896, (255, 255, 0))]:
                np.testing.assert_array_equal(frame[y, x], color)
            video.write(self.reply(), status)
            status.update(status='completed', success=True, task_completed_success_and_bucket_open=True)
            video.finish(status)
            metadata, count = self.read_video(video.path)
            self.assertEqual(metadata['size'], (1024, 864))
            self.assertEqual(metadata['fps'], 5)
            self.assertEqual(count, 12)  # Two rollout frames + two-second result.

    def test_startup_error_still_creates_result_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = self.make_video(Path(tmp) / 'evaluation.mp4')
            video.finish({'status': 'error', 'error': 'Simulator startup failed'})
            _, count = self.read_video(video.path)
            self.assertEqual(count, 10)

    def test_success_failure_and_error_have_distinct_final_captions(self):
        video = self.make_video('unused.mp4')
        success = video.render({'status': 'completed', 'success': True}, final=True)
        failure = video.render({'status': 'completed', 'success': False}, final=True)
        error = video.render({'status': 'error', 'error': 'Readback timeout'}, final=True)
        self.assertFalse(np.array_equal(success[830:], failure[830:]))
        self.assertFalse(np.array_equal(error[830:], failure[830:]))


if __name__ == '__main__':
    unittest.main()
