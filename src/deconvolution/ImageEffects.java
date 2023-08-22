package deconvolution;

import java.awt.Dimension;
import java.awt.Insets;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowEvent;
import java.awt.event.WindowListener;

import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JLabel;
import javax.swing.JSeparator;
import javax.swing.JSlider;
import javax.swing.JSpinner;
import javax.swing.SpinnerNumberModel;
import javax.swing.SwingConstants;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import javax.swing.plaf.basic.BasicArrowButton;

// This class provides a GUI for previewing and applying image effects.
// It also holds image effect settings.

public class ImageEffects {
	static final int ADJUST = 0;
	static final int SHARPEN = 1;
	static final int FAST_METHOD = 2;
	static final int RICHARDSON_LUCY = 4;
	static final int DISK_BLUR = 6;
	static final int WIENER = 7;
	
	// This stores a scaled preview image that is displayed and manipulated in edit mode
	static float[][][] previewImage;
	
	public static boolean autoPreviewEnabled = true;
	public static boolean isRendering = false;
	
	private static int option1;
	private static int option2;
	private static int option3;
	private static int option4;
	private static int option5;
	
	private static int previousOptionSum = 0; // Used to keep track of changes
	
	public static boolean isDialogShowing = false; // Whether the effect window is showing
	
	public static boolean isCanceled = false;	// Whether this effect has been canceled
	
	private static float divisor1;
	private static float divisor2;
	private static float divisor3;
	private static float divisor4;
	//private static float divisor5;
	
	public static void showEffectDialogue(final String title, final int effectType,
			String slider1Name, int min1, int max1, int default1, float divisor1,
			String slider2Name, int min2, int max2, int default2, float divisor2,
			String slider3Name, int min3, int max3, int default3, float divisor3,
			String slider4Name, int min4, int max4, int default4, float divisor4,
			String slider5Name, int min5, int max5, int default5, float divisor5) {
		
		// If there is no image, or
		// if a window is already showing, or
		// if something is rendering, then return
		if (Algorithms.imageArray == null || isDialogShowing || isRendering) {
			return;
		}
		
		isCanceled = false;
		
		// Create a reference to the image to manipulate
		previewImage = Algorithms.imageArray;
		
		// Clear out the cached image to make sure it gets updated
		Algorithms.cachedByteBuffer = null;
		
		option1 = (int)(default1 * divisor1 + 0.000001);
		option2 = (int)(default2 * divisor2 + 0.000001);
		option3 = (int)(default3 * divisor3 + 0.000001);
		option4 = (int)(default4 * divisor4 + 0.000001);
		option5 = (int)(default5 * divisor5 + 0.000001);
		
		ImageEffects.divisor1 = divisor1;
		ImageEffects.divisor2 = divisor2;
		ImageEffects.divisor3 = divisor3;
		ImageEffects.divisor4 = divisor4;
		//ImageEffects.divisor5 = divisor5;
		
		// The graphic sliders and options will be built from these arrays
		final String[] sliderNames = {slider1Name,
				slider2Name, slider3Name, slider4Name, slider5Name};
		final int[] minValues = {min1, min2, min3, min4, min5};
		final int[] maxValues = {max1, max2, max3, max4, max5};
		final int[] defaults = {default1, default2, default3, default4, default5};
		final float[] divisors = {divisor1, divisor2, divisor3, divisor4, divisor5};
		
		// Count the number of options for this graphic
		int numSliders = 0;
		for (int i = 0; i < sliderNames.length; i++) {
			if (sliderNames[i] != null) {
				numSliders++;
			}
		}
		
		// Calculate the height of the graphic window
		final int windowHeight = 70 + 62 * numSliders;
		final int windowWidth = 394;
		
		final JDialog frame = new JDialog(Interface.frame, title, JDialog.ModalityType.MODELESS);
		
		frame.getContentPane().setPreferredSize(new Dimension(windowWidth, windowHeight));
		frame.pack();
		frame.setLayout(null);
		frame.setResizable(false);
		frame.setLocationRelativeTo(Interface.frame);
		
		// This is only called when the user clicks the X button
		frame.addWindowListener(new WindowListener() {
			public void windowClosing(WindowEvent e) {
				isDialogShowing = false;
				
				// Draw the unmodified image back
				Interface.previewImage = Algorithms.arrayToImage(Algorithms.imageArray);
				Interface.lastKernel = null;
				Interface.redrawPreviewImage();
				frame.dispose();
				
				previewImage = null;
				isCanceled = true;
				Algorithms.cachedByteBuffer = null;
				
				if (!isRendering) {
					GPUAlgorithms.deallocateMemory();
				}
			}
			
			public void windowActivated(WindowEvent arg0) {}
			public void windowClosed(WindowEvent arg0) {}
			public void windowDeactivated(WindowEvent arg0) {}
			public void windowDeiconified(WindowEvent arg0) {}
			public void windowIconified(WindowEvent arg0) {}
			public void windowOpened(WindowEvent arg0) {}
		});
		
		JButton applyButton = new JButton("Apply");
		applyButton.setBounds(155, windowHeight - 34, 80, 25);
		applyButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				// Start this all in a new thread
				new Thread(new Runnable() {
					public void run() {
						renderEffect(effectType, true);
						
						// Close the frame
						frame.dispose();
						Interface.updateProgress(1);
						isDialogShowing = false;
						previewImage = null;
						
						if (!isRendering) {
							GPUAlgorithms.deallocateMemory();
						}
					}
				}).start();
			}
		});
		frame.add(applyButton);
		
		JButton cancelButton = new JButton("Cancel");
		cancelButton.setBounds(250, windowHeight - 34, 80, 25);
		cancelButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				// Draw the unmodified image back
				Interface.previewImage = Algorithms.arrayToImage(Algorithms.imageArray);
				Interface.lastKernel = null;
				Interface.redrawPreviewImage();
				frame.dispose();

				isDialogShowing = false;
				previewImage = null;
				isCanceled = true;
				
				if (!isRendering) {
					GPUAlgorithms.deallocateMemory();
				}
				
				// Special case for OpenGL:
				// Effects are never applied, so if we just closed the dialog, then just re-display the original image.
				Interface.displayOriginalPreviewOpenGL();
			}
		});
		frame.add(cancelButton);
		
		JButton previewButton = new JButton("Preview");
		previewButton.setMargin(new Insets(0, 0, 0, 0));
		previewButton.setBounds(60, windowHeight - 34, 80, 25);
		previewButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				previousOptionSum++; // Change this to prevent an early return
				renderEffect(effectType, false);
			}
		});
		frame.add(previewButton);
		
		final JSeparator bottomSep = new JSeparator(SwingConstants.HORIZONTAL);
		bottomSep.setBounds(12, windowHeight - 42, 368, 10);
		frame.add(bottomSep);
		
		int currentY = 5;
		
		// Create all of the effect parameters and sliders
		for (int i = 0; i < numSliders; i++) {
			int wordLength = sliderNames[i].length() * 7;
			
			// If there is no freedom in its values, then gray out the option
			boolean isEnabled = (minValues[i] < maxValues[i]);
			
			final JLabel label = new JLabel(sliderNames[i]);
			label.setBounds(15, currentY, wordLength, 20);
			label.setEnabled(isEnabled);
			frame.add(label);
			
			final JSeparator sep = new JSeparator(SwingConstants.HORIZONTAL);
			sep.setBounds(wordLength + 15, currentY + 10, 364 - wordLength, 10);
			frame.add(sep);
			
			// Attempt to read the radius from the file name, if it exists
			if (Interface.fileName != null && sliderNames[i].equals("Radius")) {
				int radiusTextPos = Interface.fileName.indexOf("Radius");
				if (radiusTextPos != -1) {
					int radiusStringStartIndex = radiusTextPos + 7;
					if (radiusStringStartIndex < Interface.fileName.length()) {
						int endIndex1 = Interface.fileName.indexOf(' ', radiusStringStartIndex);
						int endIndex2 = Interface.fileName.indexOf('.', radiusStringStartIndex);
						if (endIndex1 == -1) {
							endIndex1 = 99999;
						}
						if (endIndex2 == -1) {
							endIndex2 = 99999;
						}
						int minEndIndex = Math.min(endIndex1, endIndex2);
						if (minEndIndex < 99999) {
							String radiusText = Interface.fileName.substring(radiusStringStartIndex, minEndIndex);
							try {
								int newDefault = Integer.parseInt(radiusText);
								if (newDefault >= minValues[i] && newDefault < maxValues[i]) {
									defaults[i] = newDefault;
									if (i == 0) {
										option1 = (int)(defaults[i] * divisors[i] + 0.000001);
									} else if (i == 1) {
										option2 = (int)(defaults[i] * divisors[i] + 0.000001);
									} else if (i == 1) {
										option3 = (int)(defaults[i] * divisors[i] + 0.000001);
									} else if (i == 1) {
										option4 = (int)(defaults[i] * divisors[i] + 0.000001);
									} else {
										option5 = (int)(defaults[i] * divisors[i] + 0.000001);
									}
								}
							} catch (Exception e) {}
						}
					}
				}
			}
			
			final JSlider slider = new JSlider(
					JSlider.HORIZONTAL, (int)(minValues[i] * divisors[i]),
					(int)(maxValues[i] * divisors[i]), (int)(defaults[i] * divisors[i]));

			slider.setEnabled(isEnabled);
			
			final int optionNum = i+1; // Workaround for Java 7 compiler
			
			final SpinnerNumberModel spinnerModel = new SpinnerNumberModel(
					defaults[i], minValues[i], maxValues[i], 1f / divisors[i]);
			final JSpinner spinner = new JSpinner(spinnerModel);
			spinner.setBounds(300, currentY + 20, 60, 22);
			spinner.setEnabled(isEnabled);
			spinner.addChangeListener(new ChangeListener() {
				public void stateChanged(ChangeEvent e) {
					final int newValue;
					if (spinner.getValue() instanceof Float) {
						newValue = (int)((float)spinner.getValue() * divisors[optionNum-1]);
					} else {
						newValue = (int)((double)spinner.getValue() * divisors[optionNum-1]);
					}
					
					slider.setValue(newValue);
					
					if (optionNum == 1) {
						option1 = newValue;
					} else if (optionNum == 2) {
						option2 = newValue;
					} else if (optionNum == 3) {
						option3 = newValue;
					} else if (optionNum == 4) {
						option4 = newValue;
					} else if (optionNum == 5) {
						option5 = newValue;
					}
					
					if (autoPreviewEnabled) {
						renderEffect(effectType, false);
					}
				}
			});
			frame.add(spinner);
			
			final int dist = maxValues[i] - minValues[i];
			int majorSpacing = (int)(dist / 40 * divisors[i]) * 10;
			if (majorSpacing < 1) {
				majorSpacing = 1;
				if (majorSpacing < dist * divisors[i] / 5) {
					majorSpacing = (int)(dist * divisors[i] / 5 + 0.01);
				}
			}
			
			slider.setBounds(20, currentY + 18, 270, 50);
			if (divisors[i] == 1) {
				slider.setPaintLabels(true);
			}
			slider.setPaintTicks(true);
			slider.setOpaque(false);
			slider.setMajorTickSpacing(majorSpacing);
			slider.addChangeListener(new ChangeListener() {
				public void stateChanged(ChangeEvent e) {
					spinner.setValue(slider.getValue() / divisors[optionNum - 1]);
					
					if (optionNum == 1) {
						option1 = slider.getValue();
					} else if (optionNum == 2) {
						option2 = slider.getValue();
					} else if (optionNum == 3) {
						option3 = slider.getValue();
					} else if (optionNum == 4) {
						option4 = slider.getValue();
					} else if (optionNum == 5) {
						option5 = slider.getValue();
					}
					
					if (autoPreviewEnabled) {
						renderEffect(effectType, false);
					}
				}
			});
			frame.add(slider);
			
			// Add the reset do default button for each slider
			final BasicArrowButton resetButton = new BasicArrowButton(BasicArrowButton.WEST);
			resetButton.setEnabled(isEnabled);
			resetButton.addActionListener(new ActionListener() {
				public void actionPerformed(ActionEvent e) {
					if (spinner.getValue() instanceof Float) {
						spinner.setValue((float)defaults[optionNum-1]);
					} else if (spinner.getValue() instanceof Double) {
						spinner.setValue((double)defaults[optionNum-1]);
					} else if (spinner.getValue() instanceof Integer) {
						spinner.setValue((int)defaults[optionNum-1]);
					}
					
				}
			});
			resetButton.setBounds(365, currentY + 20, 22, 22);
			frame.add(resetButton);
			
			currentY += 65;
		}

		isDialogShowing = true;
		frame.setVisible(true);
	}
	
	private static void renderEffect(final int effectType, final boolean commit) {
		
		if (previewImage == null || isRendering) {
			return;
		}
		
		// Return if nothing changed
		if (!commit && option1 + option2 + option3 + option4 + option5 == previousOptionSum) {
			return;
		}
		
		// Record this for later
		previousOptionSum = option1 + option2 + option3 + option4 + option5;
		
		isRendering = true;
		
		// Start this all in a new thread
		new Thread(new Runnable() {
			public void run() {
				
				// Store the starting options to detect if there was a change while rendering
				final int tempOption1 = option1;
				final int tempOption2 = option2;
				final int tempOption3 = option3;
				final int tempOption4 = option4;
				final int tempOption5 = option5;
				
				float[][][] newImageArray = null;
				
				if (effectType == ADJUST) {
					newImageArray = Algorithms.adjust(
							previewImage, option1 / divisor1,
							option2 / divisor2, option3 / divisor3,
							option4 / divisor4 * 0.12f + 1);
				} else if (effectType == SHARPEN) {
					newImageArray = Algorithms.sharpen(
							previewImage, option1 / divisor1 / 100f, option2 / divisor2);
				} else if (effectType == FAST_METHOD) {
					newImageArray = Algorithms.fastMethodSwitch(
							previewImage, option1 / divisor1, option2 / divisor2,
							(int)(option3 / divisor3), commit);
				} else if (effectType == RICHARDSON_LUCY) {
					newImageArray = Algorithms.richardsonLucySwitch(
							previewImage, option1 / divisor1, (int)(option2 / divisor2), commit);
				} else if (effectType == WIENER) {
					newImageArray = WienerFilter.wienerDeconvolvePublic(
							previewImage, (int)(option1 / divisor1), (int)(option2 / divisor2));
				} else if (effectType == DISK_BLUR) {
					newImageArray = Algorithms.diskBlur(
							previewImage, option1 / divisor1);
				} else {
					System.err.println("Effect not set");
					isRendering = false;
					return;
				}
				
				assert commit && newImageArray == null : "Cannot commit null image!";
				
				// Deallocate the GPU memory if the preview has ended
				if (!ImageEffects.isDialogShowing) {
					GPUAlgorithms.deallocateMemory();
				}
				
				if (isCanceled) {
					Interface.cancelProgress();
					Interface.previewImage = Algorithms.arrayToImage(Algorithms.imageArray);
				} else {
					// This may be null if the GPU has rendered data directly to the BufferedImage
					if (newImageArray != null) {
						if (commit) {
							Algorithms.imageArray = newImageArray;
							Interface.lastKernel = null;
						}
						Interface.previewImage = Algorithms.arrayToImage(newImageArray);
					}
				}
				Interface.redrawPreviewImage();
				
				isRendering = false;
				
				// If the preview should be rendered automatically
				if (autoPreviewEnabled && !commit &&
							(tempOption1 != option1 || tempOption2 != option2 ||
							tempOption3 != option3 || tempOption4 != option4 ||
							tempOption5 != option5)) {
					// The settings have changed,
					//   so render this same effect again
					renderEffect(effectType, false);
				}
				
				// Special case for OpenGL:
				// Effects are never applied, so if we just closed the dialog, then just re-display the original image.
				if (commit) {
					Interface.displayOriginalPreviewOpenGL();
				}
			}
		}).start();
	}
	
	static void print(Object o) {
		System.out.println(o);
	}
}
