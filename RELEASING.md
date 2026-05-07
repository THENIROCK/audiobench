# Releasing audiobench

This is the maintainer playbook for cutting a new release. The whole flow
is: bump the version, write the changelog entry, tag, push, build, upload
to TestPyPI, smoke-test, upload to PyPI, publish a GitHub Release.

## One-time setup

You only do this once per machine.

### 1. PyPI and TestPyPI accounts

Create accounts at:

- <https://pypi.org/account/register/>
- <https://test.pypi.org/account/register/>

These are **two separate accounts** with two separate passwords. PyPI
requires 2FA for new uploads — enable it on both.

### 2. API tokens

Generate one token per account:

- <https://pypi.org/manage/account/token/>
- <https://test.pypi.org/manage/account/token/>

For the first upload, scope the token to "Entire account". After the
first successful upload of `audiobench`, you can re-scope to "Project:
audiobench" only.

### 3. `~/.pypirc`

Drop both tokens into `~/.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
  username = __token__
  password = pypi-AgEI...your-pypi-token...

[testpypi]
  repository = https://test.pypi.org/legacy/
  username = __token__
  password = pypi-AgEI...your-testpypi-token...
```

The `[pypi]` section does **not** need a `repository` line — twine's
default upload URL is `https://upload.pypi.org/legacy/`, which is real
PyPI. The `repository` line is required under `[testpypi]` because
TestPyPI is not the default.

Lock down the file:

```bash
chmod 600 ~/.pypirc
```

### 4. Install build tools in your venv

```bash
.venv/bin/pip install --upgrade build twine
```

These are also available as a `[dev]` extra:

```bash
.venv/bin/pip install -e ".[dev]"
```

### 5. GitHub CLI (for cutting GitHub Releases)

```bash
brew install gh
gh auth login
```

Pick: GitHub.com → HTTPS → Yes (authenticate Git too) → Login with a web
browser → paste the one-time code → click Authorize.

Verify:

```bash
gh auth status
# Should print: Logged in to github.com account THENIROCK
```

## Per-release flow

For each release, work through these steps in order. The version-bump
choice is up to you (semver: patch / minor / major).

### 1. Update `CHANGELOG.md`

Move the bullets from the `[Unreleased]` section into a new dated section
above it. Update the link references at the bottom.

```markdown
## [Unreleased]

## [0.2.0] - YYYY-MM-DD

### Added
- ...

### Changed
- ...

### Fixed
- ...

[Unreleased]: https://github.com/THENIROCK/audiobench/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/THENIROCK/audiobench/releases/tag/v0.2.0
[0.1.1]: https://github.com/THENIROCK/audiobench/releases/tag/v0.1.1
```

### 2. Bump the version in `pyproject.toml`

```toml
[project]
name = "audiobench"
version = "0.2.0"
```

### 3. Commit and tag

```bash
git add CHANGELOG.md pyproject.toml
git commit -m "Release 0.2.0: <one-line summary>"

git tag -a v0.2.0 -F - <<'EOF'
audiobench 0.2.0

<one-paragraph summary>

Highlights:
- ...
- ...
EOF

git push && git push origin v0.2.0
```

Why annotated tags (`-a` / `-F`) instead of lightweight (`git tag v0.2.0`):
annotated tags carry tagger, date, and a message, and they're what
GitHub Releases, `gh`, and PyPI's "Release notes" tooling all expect.

### 4. Build the sdist + wheel

```bash
rm -rf dist/ build/ && rm -rf src/audiobench.egg-info
.venv/bin/python -m build
.venv/bin/twine check dist/*
```

Sanity-check that data files are bundled:

```bash
unzip -l dist/audiobench-0.2.0-py3-none-any.whl | \
  grep -E "(LICENSE|prompts.yaml|packs/.*json|clips/.*wav|manifest.json)"
```

You're looking for: `audiobench/data/sound_id/packs/*.json`,
`prompts.yaml`, the `asr_robust/clips/*.wav`, `manifest.json`, and the
bundled LICENSE.

### 5. Upload to TestPyPI first

Always smoke-test on TestPyPI before real PyPI. PyPI versions are
**immutable** — you cannot reupload `0.2.0` after pulling it, even if
you yank it. TestPyPI is your sandbox.

```bash
.venv/bin/twine upload --repository testpypi dist/*
```

Verify in a clean venv:

```bash
python -m venv /tmp/abtest && source /tmp/abtest/bin/activate
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            audiobench
audiobench --help
audiobench run ab/sound-id --profile demo-fast --model heuristic-v0
deactivate && rm -rf /tmp/abtest
```

The `--extra-index-url` is required because TestPyPI does not host all
your dependencies — pip needs the real PyPI for those.

### 6. Upload to real PyPI

```bash
.venv/bin/twine upload dist/*
```

Final smoke test from the real index:

```bash
python -m venv /tmp/abreal && source /tmp/abreal/bin/activate
pip install audiobench
audiobench --help
deactivate && rm -rf /tmp/abreal
```

The PyPI project page should now be live at
<https://pypi.org/project/audiobench/>.

### 7. Publish the GitHub Release

Extract the changelog section for this version into a notes file:

```bash
awk '/^## \[0\.2\.0\]/{flag=1; next} /^## \[/{flag=0} flag' CHANGELOG.md \
  > /tmp/audiobench-0.2.0-notes.md
```

Create the release with the wheel and sdist attached:

```bash
gh release create v0.2.0 \
  --title "audiobench 0.2.0" \
  --notes-file /tmp/audiobench-0.2.0-notes.md \
  --verify-tag

gh release upload v0.2.0 \
  dist/audiobench-0.2.0.tar.gz \
  dist/audiobench-0.2.0-py3-none-any.whl
```

`--verify-tag` makes `gh` refuse to create the release if the tag
doesn't exist on the remote yet — guards against typos.

## Known gotchas (real things we've hit)

### macOS: `ModuleNotFoundError: No module named 'audiobench'` after `pip install -e .`

Symptom: `audiobench --help` raises `ModuleNotFoundError` immediately
after a successful editable install.

Cause: pip-installed files inherit a `com.apple.provenance` extended
attribute that carries the `UF_HIDDEN` BSD flag. Python 3.13's
`site.py` skips `.pth` files with that flag, so the editable-install
pointer never lands on `sys.path`.

This is a known interaction documented in:
- Python issue [#127012](https://github.com/python/cpython/issues/127012)
- pip issue [#13153](https://github.com/pypa/pip/issues/13153)

Fix:

```bash
chflags -R nohidden .venv/lib/python3.13/site-packages
```

The flag re-attaches every time pip writes to `site-packages`, so
re-run after each `pip install` if the bug bites again. Long-term, `uv`
(`uv venv && uv pip install -e .`) doesn't carry the flag forward.

### macOS: `gh auth login` fails with `permission denied` on `~/.config/gh/config.yml`

Symptom: `gh` insists on sudo to do anything; the underlying error is
`permission denied` reading or writing `~/.config/gh/config.yml`.

Cause: `~/.config` is owned by root because something in your past was
run with `sudo` and created the directory while elevated.

Fix (the only legitimate sudo here):

```bash
sudo chown -R "$(whoami):staff" ~/.config
```

After that, `gh auth login` works without sudo.

### PyPI: "File already exists" when uploading

Symptom: `twine upload` returns an HTTP 400 saying the version already
exists on PyPI.

Cause: PyPI versions are immutable. Even if you yank a release you
cannot reupload the same filename or version.

Fix: bump the version in `pyproject.toml` (and add a new `CHANGELOG.md`
entry), rebuild, then re-upload.

### Build: data files missing from the wheel

Symptom: `import audiobench` works after `pip install`, but commands
like `audiobench run ab/sound-id` fail with "manifest.json not found".

Cause: `[tool.setuptools.package-data]` glob doesn't match the file.

Fix: confirm the path is correct in `pyproject.toml`:

```toml
[tool.setuptools.package-data]
"audiobench.data.asr_robust" = ["manifest.json", "clips/*.wav"]
"audiobench.data.sound_id"   = ["packs/*.json", "prompts.yaml"]
```

Verify what's actually in the wheel:

```bash
unzip -l dist/audiobench-*.whl | grep data/
```

## Quick reference: full release in one block

```bash
# 1. Update CHANGELOG.md and bump version in pyproject.toml.

# 2. Commit + tag + push.
VERSION=0.2.0
git add CHANGELOG.md pyproject.toml
git commit -m "Release ${VERSION}: <summary>"
git tag -a "v${VERSION}" -F - <<EOF
audiobench ${VERSION}

<summary>
EOF
git push && git push origin "v${VERSION}"

# 3. Build.
rm -rf dist/ build/ src/audiobench.egg-info
.venv/bin/python -m build
.venv/bin/twine check dist/*

# 4. TestPyPI smoke.
.venv/bin/twine upload --repository testpypi dist/*
python -m venv /tmp/abtest && source /tmp/abtest/bin/activate
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            audiobench
audiobench --help
deactivate && rm -rf /tmp/abtest

# 5. Real PyPI.
.venv/bin/twine upload dist/*

# 6. GitHub Release.
awk -v v="$VERSION" '$0 ~ "^## \\[" v "\\]"{flag=1; next} /^## \[/{flag=0} flag' \
  CHANGELOG.md > "/tmp/audiobench-${VERSION}-notes.md"
gh release create "v${VERSION}" \
  --title "audiobench ${VERSION}" \
  --notes-file "/tmp/audiobench-${VERSION}-notes.md" \
  --verify-tag
gh release upload "v${VERSION}" \
  "dist/audiobench-${VERSION}.tar.gz" \
  "dist/audiobench-${VERSION}-py3-none-any.whl"
```

## Future improvement: Trusted Publishers

Once you've done a couple of manual releases, swap the API token flow
for [PyPI Trusted Publishers](https://docs.pypi.org/trusted-publishers/)
via GitHub Actions. Trusted Publishing uses GitHub's OIDC tokens to
vouch for the repo, so you never store a long-lived PyPI API token
anywhere. The whole release becomes "push a tag, watch CI publish it".
