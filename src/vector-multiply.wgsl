@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> parameters: vec4<u32>;

override workgroupSize: u32 = 64u;

@compute @workgroup_size(workgroupSize)
fn multiply(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index < arrayLength(&output)) {
    output[index] = left[index] * right[index];
  }
}

@compute @workgroup_size(workgroupSize)
fn add(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index < arrayLength(&output)) {
    output[index] = left[index] + right[index];
  }
}

@compute @workgroup_size(workgroupSize)
fn scale(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index < arrayLength(&output)) {
    output[index] = left[index] * bitcast<f32>(parameters.x);
  }
}
