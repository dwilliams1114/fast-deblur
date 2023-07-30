// Developed along with DeblurOpenGL.java and fshader.glsl

#version 120

void main()
{
	gl_TexCoord[0] = gl_MultiTexCoord0;
	gl_Position = ftransform();
}