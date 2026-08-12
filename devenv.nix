{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:

{
  packages = with pkgs; [
    git
    python313Packages.tensorflow
    python313Packages.numpy
  ];

  languages.python = {
    enable = true;
    version = "3.13";
    venv.enable = true;
    venv.requirements = ''
      maze-utils
    '';
  };
}
