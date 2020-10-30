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

pub mod imgui_vs {
    vulkano_shaders::shader!{
        ty: "vertex",
        src: "
#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 position;
layout(location = 1) in vec2 UV;
layout(location = 2) in vec4 color;

layout(push_constant) uniform Matrices {
    mat4 ortho;
} matrices;

layout(location = 0) out vec4 oColor;
layout(location = 1) out vec2 oUV;

void main() {
    oColor = color;
    oUV = UV;

    gl_Position = matrices.ortho*vec4(position.x, position.y, 0.0, 1.0);
}
"
    }
}

pub mod imgui_fs {
    vulkano_shaders::shader!{
        ty: "fragment",
        src: "
#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec4 color;
layout(location = 1) in vec2 oUV;

layout(binding = 0, set = 0) uniform sampler2D fontsSampler;

layout(location = 0) out vec4 finalColor;

void main() {
    finalColor = color * texture(fontsSampler, oUV);
    // finalColor = color;
}
"
    }
}