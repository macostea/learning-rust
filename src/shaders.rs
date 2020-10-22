pub mod vs {
    vulkano_shaders::shader!{
        ty: "vertex",
        src: "
#version 450

layout(location = 0) in vec4 position;
layout(location = 1) in vec4 color;
layout(set = 0, binding = 0) uniform Data {
     mat4 perspectiveMatrix;
     vec2 offset;
} uniforms;

layout(location = 0) out vec4 fragColor;


void main() {
    vec4 cameraPos = position + vec4(uniforms.offset.x, uniforms.offset.y, 0.0, 0.0);

    gl_Position = uniforms.perspectiveMatrix * cameraPos;

    fragColor = color;
}"
    }
}

pub mod fs {
    vulkano_shaders::shader!{
        ty: "fragment",
        src: "
#version 450

layout(location = 0) in vec4 fragColor;

layout(location = 0) out vec4 f_color;

void main() {
    f_color = fragColor;
}"
    }
}

