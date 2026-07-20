-- Actual render-camera bridge for ac_dataset.
-- No image data is read and no pixel-based calibration is performed.

local CAPACITY = 128
local MAX_CARS = 32
local MAGIC = 0x41434350 -- "ACCP"

local mmf = ac.writeMemoryMappedFile('AcTools.ACDataset.RenderState.v2', [[
  uint32_t magic;
  uint32_t version;
  uint32_t capacity;
  uint32_t write_index;
  uint32_t slot_seq[128];
  int32_t render_frame[128];
  float sim_time_ms[128];
  int32_t lap_time_ms[128];
  int32_t lap_count[128];
  float pos_x[128];
  float pos_y[128];
  float pos_z[128];
  float forward_x[128];
  float forward_y[128];
  float forward_z[128];
  float up_x[128];
  float up_y[128];
  float up_z[128];
  float fov_deg[128];
  int32_t camera_mode[128];
  int32_t drivable_camera_mode[128];
  int32_t car_camera_index[128];
  int32_t focused_car[128];
  int32_t car_count[128];
  float car_timestamp_ms[4096];
  float car_pos_x[4096];
  float car_pos_y[4096];
  float car_pos_z[4096];
  float car_forward_x[4096];
  float car_forward_y[4096];
  float car_forward_z[4096];
  float car_up_x[4096];
  float car_up_y[4096];
  float car_up_z[4096];
  float car_aabb_center_forward[4096];
  float car_aabb_center_right[4096];
  float car_aabb_center_up[4096];
  float car_aabb_length[4096];
  float car_aabb_width[4096];
  float car_aabb_height[4096];
]], false)

mmf.magic = MAGIC
mmf.version = 2
mmf.capacity = CAPACITY
mmf.write_index = 0

function script.update(dt)
  local sim = ac.getSim()
  local car = ac.getCar(0)
  if car == nil then return end

  local n = mmf.write_index
  local i = n % CAPACITY
  local seq = n * 2 + 1
  mmf.slot_seq[i] = seq -- odd: writer owns this slot

  local p = ac.getCameraPosition()
  local f = ac.getCameraForward()
  local u = ac.getCameraUp()
  mmf.render_frame[i] = sim.frame
  mmf.sim_time_ms[i] = sim.time
  mmf.lap_time_ms[i] = car.lapTimeMs
  mmf.lap_count[i] = car.lapCount

  -- Dataset/AC Python shared-memory convention: (x,y,z) = (AC z, AC x, AC y).
  mmf.pos_x[i], mmf.pos_y[i], mmf.pos_z[i] = p.z, p.x, p.y
  mmf.forward_x[i], mmf.forward_y[i], mmf.forward_z[i] = f.z, f.x, f.y
  mmf.up_x[i], mmf.up_y[i], mmf.up_z[i] = u.z, u.x, u.y
  mmf.fov_deg[i] = ac.getCameraFOV()
  mmf.camera_mode[i] = sim.cameraMode
  mmf.drivable_camera_mode[i] = sim.driveableCameraMode
  mmf.car_camera_index[i] = sim.carCameraIndex
  mmf.focused_car[i] = sim.focusedCar

  -- Camera and cars are sampled in this same render callback. This removes the
  -- asynchronous Python opponent-server delay from 3D box projection.
  local count = math.min(sim.carsCount, MAX_CARS)
  mmf.car_count[i] = count
  for carID = 0, count - 1 do
    local c = ac.getCar(carID)
    if c ~= nil then
      local k = i * MAX_CARS + carID
      local cp, cf, cu = c.position, c.look, c.up
      local bc, bs = c.aabbCenter, c.aabbSize
      mmf.car_timestamp_ms[k] = c.timestamp
      mmf.car_pos_x[k], mmf.car_pos_y[k], mmf.car_pos_z[k] = cp.z, cp.x, cp.y
      mmf.car_forward_x[k], mmf.car_forward_y[k], mmf.car_forward_z[k] = cf.z, cf.x, cf.y
      mmf.car_up_x[k], mmf.car_up_y[k], mmf.car_up_z[k] = cu.z, cu.x, cu.y
      mmf.car_aabb_center_forward[k] = bc.z
      mmf.car_aabb_center_right[k] = bc.x
      mmf.car_aabb_center_up[k] = bc.y
      mmf.car_aabb_length[k], mmf.car_aabb_width[k], mmf.car_aabb_height[k] = bs.z, bs.x, bs.y
    end
  end

  mmf.slot_seq[i] = seq + 1 -- even: complete sample
  mmf.write_index = n + 1
end

function script.windowMain()
  ui.text('Actual render camera pose bridge is active.')
  ui.text(string.format('Render frame: %d  Samples: %d  Cars: %d',
    ac.getSim().frame, mmf.write_index, ac.getSim().carsCount))
  ui.text('No image pixels are read or used for calibration.')
end
