# Dagster video media image

The video job needs `ffmpeg` and `ffprobe`. Dagster+ Serverless runs this repository as a Python executable on a base image. Its default image does not guarantee those tools. The video schedule stays stopped until the image and the rest of the workflow pass the activation gate.

Run the **Build Dagster video base image** workflow from `main`. It builds the image, encodes and probes a fixture, exercises loudness normalization, and uploads the image to Dagster+. The run summary reports the exact `video-ffmpeg-<commit>` tag.

Set repository variable `SERVERLESS_BASE_IMAGE_TAG` to that reported tag. The next push to `main` deploys with the tagged image. Confirm the deployed code location uses that tag and execute the same media smoke in an isolated production run before enabling the video schedule. Keep the tag in the activation record so the image can be reproduced or rolled back.

The custom base image process follows the [Dagster+ Serverless native dependency documentation](https://master.dagster.dagster-docs.io/dagster-plus/deployment/serverless#using-a-different-base-image-or-using-native-dependencies).
