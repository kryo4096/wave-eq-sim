use std::sync::Arc;

use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBuffer},
    descriptor::{descriptor_set::PersistentDescriptorSet, PipelineLayoutAbstract},
    device::{Device, DeviceExtensions, Features},
    instance::{Instance, InstanceExtensions, PhysicalDevice, QueueFamily},
    pipeline::{shader::SpecializationConstants, ComputePipeline},
    sync::GpuFuture,
};

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "res/shaders/compute.glsl"
    }
}

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
        &DeviceExtensions {
            khr_storage_buffer_storage_class: true,
            ..DeviceExtensions::none()
        },
        std::array::IntoIter::new([(queue_family, 0.5)]),
    )
    .expect("failed to create device");

    let queue = queues.next().unwrap();

    let data_buffer =
        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, 0..65536)
            .expect("failed to create buffer");

    let shader = cs::Shader::load(device.clone()).expect("failed to create shader");

    let compute_pipeline = Arc::new(
        ComputePipeline::new(device.clone(), &shader.main_entry_point(), &(), None)
            .expect("failed to create compute pipeline"),
    );

    let layout = compute_pipeline.layout().descriptor_set_layout(0).unwrap();

    let set = Arc::new(
        PersistentDescriptorSet::start(layout.clone())
            .add_buffer(data_buffer.clone())
            .unwrap()
            .build()
            .unwrap(),
    );

    let mut builder = AutoCommandBufferBuilder::new(device.clone(), queue.family()).unwrap();
    builder
        .dispatch(
            [1024, 1, 1],
            compute_pipeline.clone(),
            set.clone(),
            (),
            None,
        )
        .unwrap();

    let command_buffer = builder.build().unwrap();

    let finished = command_buffer.execute(queue.clone()).unwrap();

    finished
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    let content = data_buffer.read().unwrap();
    for (n, val) in content.iter().enumerate() {
        assert_eq!(*val, n as u32 * 12);
    }

    println!("Everything succeeded!");
}
