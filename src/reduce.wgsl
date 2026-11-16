@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;

// 0 = sum, 1 = min, 2 = max. The latter two compare sortable IEEE 754 bit patterns.
override mode: u32;
const sign = 0x80000000u;
const inf = 0x7f800000u;
const nan = 0x7fc00000u;
var<workgroup> values: array<u32, 64>;
var<workgroup> flags: array<u32, 64>;

@compute @workgroup_size(64)
fn reduce(@builtin(global_invocation_id) global: vec3<u32>, @builtin(local_invocation_index) lane: u32,
    @builtin(workgroup_id) group: vec3<u32>) {
  var value = select(0u, 0xffffffffu, mode == 1u);
  var flag = 0u;
  if (global.x < arrayLength(&input)) {
    let bits = input[global.x];
    let magnitude = bits & ~sign;
    if (magnitude > inf) {
      flag = 1u;
    } else if (mode == 0u) {
      if (magnitude == inf) {
        flag = select(2u, 4u, (bits & sign) != 0u);
      } else {
        // Scale so that a finite sum of 64 maximum f32 values cannot overflow on the device.
        value = bitcast<u32>(bitcast<f32>(bits) / 64.0);
      }
    } else {
      value = select(bits ^ sign, ~bits, (bits & sign) != 0u);
    }
  }
  values[lane] = value;
  flags[lane] = flag;
  workgroupBarrier();

  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (lane < stride) {
      let other = values[lane + stride];
      if (mode == 0u) {
        values[lane] = bitcast<u32>(bitcast<f32>(values[lane]) + bitcast<f32>(other));
      } else if (mode == 1u) {
        values[lane] = min(values[lane], other);
      } else {
        values[lane] = max(values[lane], other);
      }
      flags[lane] |= flags[lane + stride];
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    let tags = flags[0];
    if ((tags & 1u) != 0u || tags == 6u) {
      output[group.x] = nan;
    } else if (tags != 0u) {
      output[group.x] = select(inf, sign | inf, tags == 4u);
    } else if (mode == 0u) {
      output[group.x] = values[0];
    } else {
      output[group.x] = select(~values[0], values[0] ^ sign, (values[0] & sign) != 0u);
    }
  }
}
