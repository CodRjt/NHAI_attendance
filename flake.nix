{
  description = "React Native & C++ JSI development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            android_sdk.accept_license = true;
          };
        };

        androidComposition = pkgs.androidenv.composeAndroidPackages {
          buildToolsVersions = [
            "33.0.0"
            "34.0.0"
            "35.0.0"
            "36.0.0"
          ];
          platformVersions = [
            "34"
            "35"
            "36"
          ];
          abiVersions = [
            "armeabi-v7a"
            "arm64-v8a"
            "x86_64"
          ];

          includeNDK = true;
          ndkVersions = [ "28.2.13676358" ];

          includeCmake = true;
          cmakeVersions = [ "3.22.1" ];
        };

        androidSdk = androidComposition.androidsdk;
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            pnpm

            androidSdk
            jdk17

            cmake
            gnumake
            pkg-config
            ninja

            vtsls
            clang-tools
          ];

          ANDROID_HOME = "${androidSdk}/libexec/android-sdk";
          ANDROID_NDK_ROOT = "${androidSdk}/libexec/android-sdk/ndk/28.2.13676358";
          JAVA_HOME = "${pkgs.jdk17}/lib/openjdk";

          shellHook = ''
            export PATH="$PATH:$PWD/node_modules/.bin"
            export PATH="$PATH:${androidSdk}/libexec/android-sdk/platform-tools"
            export PATH="$PATH:${androidSdk}/libexec/android-sdk/cmdline-tools/latest/bin"
          '';
        };
      }
    );
}
