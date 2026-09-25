# Dagster video media image

The video job needs `ffmpeg` and `ffprobe`. Dagster+ Serverless runs this repository as a Python executable on a base image. Its default image does not guarantee those tools. The video schedule stays stopped until the image and the rest of the workflow pass the activation gate.

Run the **Build Dagster video base image** workflow from `main`. It builds the image, encodes and probes a fixture, exercises loudness normalization, and uploads the image to Dagster+. The run summary reports the exact `video-ffmpeg-<commit>` tag.

Set repository variable `SERVERLESS_BASE_IMAGE_TAG` to that reported tag. The next push to `main` deploys with the tagged image. Confirm the deployed code location uses that tag and execute the same media smoke in an isolated production run before enabling the video schedule. Keep the tag in the activation record so the image can be reproduced or rolled back.

On 2026-09-25, [workflow run 36124177384](https://github.com/mauricedesaxe/news-aggregator/actions/runs/36124177384) built, tested, and uploaded `video-ffmpeg-d6850082ee705806a71789bfc483b94f42f50306` with image digest `sha256:888a0f598ab9d77bd80c243a32b46760f36b6b54eb48496d77188af30cc1ae28`. The repository variable now points to that tag. Production deployment and the isolated media smoke remain to be verified before the schedule starts.

The custom base image process follows the [Dagster+ Serverless native dependency documentation](https://master.dagster.dagster-docs.io/dagster-plus/deployment/serverless#using-a-different-base-image-or-using-native-dependencies).
