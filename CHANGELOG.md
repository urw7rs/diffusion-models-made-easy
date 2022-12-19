# Changelog

## v0.1.0 (12/20/2022)

### API changes
- `LitDDPM` takes samplers as arguments instead of creating them internally.
- `DDPMSampler`sets beta values (i.e. noise schedule) internally.

### Features
- Add EMA callback
- Documentation for code and a summary of DDPM

## v0.0.2 (12/16/2022)

### Fix

- Import `make_history` in `LitDDPM`

## v0.0.1 (12/16/2022)

- First release of `dmme`
- Added DDPM
