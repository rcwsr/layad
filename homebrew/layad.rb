# typed: strict
# frozen_string_literal: true

# Formula for the rcwsr/homebrew-tap tap. Copy to the tap root (alongside treesync.rb)
# with scripts/bump-formula.sh, which fills in the version and tarball checksum.
class Layad < Formula
  include Language::Python::Virtualenv

  desc "Local daemon that keeps the Laya decision model resident and serves it over HTTP"
  homepage "https://github.com/rcwsr/layad"
  url "https://github.com/rcwsr/layad/archive/refs/tags/v0.1.0.tar.gz"
  # Placeholder: scripts/bump-formula.sh rewrites url and sha256 at release time.
  sha256 "0000000000000000000000000000000000000000000000000000000000000000"
  license "MIT"

  # MLX compiles to Metal: there is no Intel build and no Linux build, and mlx wheels
  # start at macosx_14_0_arm64.
  depends_on arch: :arm64
  depends_on macos: :sonoma
  # Pinned deliberately: requirements.lock is resolved for cp313 and mlx ships ABI-tagged
  # wheels, so the lock and this interpreter have to move together.
  depends_on "python@3.13"

  def install
    virtualenv_create(libexec, "python3.13")
    # The dependency closure includes mlx, mlx-metal, tokenizers, pydantic-core and numpy,
    # none of which build from source here (mlx publishes no sdist at all), so this installs
    # wheels. --require-hashes keeps that honest: every artifact is pinned in the lock.
    system libexec/"bin/pip", "install", "--no-cache-dir", "--require-hashes",
           "-r", "requirements.lock"
    system libexec/"bin/pip", "install", "--no-cache-dir", "--no-deps", buildpath
    bin.install_symlink libexec/"bin/layad"
  end

  service do
    run [opt_bin/"layad", "serve"]
    run_type :immediate
    keep_alive true
    environment_variables LAYAD_WARM: "1"
    working_dir Dir.home
    log_path var/"log/layad.log"
    error_log_path var/"log/layad.err.log"
  end

  def caveats
    <<~EOS
      layad can be managed by launchd in one of two ways, and only one at a time --
      both bind 127.0.0.1:8918, so running both leaves one of them crash-looping:

        brew services start layad     (label homebrew.mxcl.layad)
        layad install-agent           (label com.rcwsr.layad)

      `layad install-agent` refuses to install over a brew-managed service. Run
      `layad config` to see which one is active.

      The first request downloads the checkpoint (~800 MiB) into ~/.cache/huggingface
      and takes roughly 11.6 s to load; warm calls are ~11 ms.
    EOS
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/layad --version")
    # Deliberately no inference here: brew test must not pull an 800 MiB checkpoint.
    assert_match "127.0.0.1", shell_output("#{bin}/layad config")
  end
end
