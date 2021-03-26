use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBuffer},
    device::{Device, DeviceExtensions, Features},
    instance::{Instance, InstanceExtensions, PhysicalDevice, QueueFamily},
    sync::GpuFuture,
};

fn main() {
    let instance =
        Instance::new(None, &InstanceExtensions::none(), None).expect("failed to create instance");

    let physical = PhysicalDevice::enumerate(&instance)
        .next()
        .expect("no device available");

    let queue_family = physical
        .queue_families()
        .find(QueueFamily::supports_graphics)
        .expect("couldn't find a graphical queue family");

    let (device, mut queues) = Device::new(
        physical,
        &Features::none(),
        &DeviceExtensions::none(),
        std::array::IntoIter::new([(queue_family, 0.5)]),
    )
    .expect("failed to create device");

    let queue = queues.next().unwrap();

    let source_buffer =
        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, 0..64)
            .expect("failed to create buffer");

    let dest_buffer = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::all(),
        false,
        (0..64).map(|_| 0),
    )
    .expect("failed to create buffer");

    let mut builder = AutoCommandBufferBuilder::new(device.clone(), queue.family()).unwrap();

    builder
        .copy_buffer(source_buffer.clone(), dest_buffer.clone())
        .unwrap();

    let command_buffer = builder.build().unwrap();

    let finished = command_buffer.execute(queue.clone()).unwrap();

    finished
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    let src_content = source_buffer.read().unwrap();
    let dest_content = dest_buffer.read().unwrap();
    assert_eq!(&*src_content, &*dest_content);
}
