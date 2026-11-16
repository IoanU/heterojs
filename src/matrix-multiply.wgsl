@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> dimensions: vec4<u32>;

const tileSize = 8u;
var<workgroup> leftTile: array<array<f32, 8>, 8>;
var<workgroup> rightTile: array<array<f32, 8>, 8>;

@compute @workgroup_size(8, 8)
fn matrix(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
  let rows = dimensions.x;
  let inner = dimensions.y;
  let columns = dimensions.z;
  let row = id.y;
  let col = id.x;
  var value = 0.0;
  for (var start = 0u; start < inner; start += tileSize) {
    let leftCol = start + local.x;
    let rightRow = start + local.y;
    var a = 0.0;
    var b = 0.0;
    if (row < rows && leftCol < inner) {
      a = left[row * inner + leftCol];
    }
    if (rightRow < inner && col < columns) {
      b = right[rightRow * columns + col];
    }
    leftTile[local.y][local.x] = a;
    rightTile[local.y][local.x] = b;
    workgroupBarrier();

    for (var k = 0u; k < tileSize; k++) {
      value += leftTile[local.y][k] * rightTile[k][local.x];
    }
    workgroupBarrier();
  }
  if (row < rows && col < columns) {
    output[row * columns + col] = value;
  }
}
