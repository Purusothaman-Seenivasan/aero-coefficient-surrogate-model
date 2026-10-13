README.pdf: README.md preamble.tex
	pandoc README.md -o README.pdf \
		--standalone \
		--pdf-engine=xelatex \
		-V mainfont="Bitstream Charter" \
		-V monofont="FiraCode Nerd Font Mono" \
		-V fontfamily=mathpazo \
		-V fontsize=11pt \
		-V geometry:margin=0.75in \
		-V colorlinks \
		-V urlcolor=RoyalBlue \
		-V linkcolor=RoyalBlue \
		--include-in-header=preamble.tex \
		--highlight-style=tango \
		--metadata title="HiLiftAeroML Splits README"

.PHONY: pdf clean
pdf: README.pdf
clean:
	rm -f README.pdf