%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Class file for the paper of the Bicycle and Motercycle Dynamics
%% Conference 2023, 18-20 September, Delft University of Technology.
%% Jason K. Moore & Christoph Schmidt 20230110 revision
%% Ben Smith 10092015 REV 2018 by UNIPD team
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{bmd2023p}[2023/05/20 BMD2023 paper class]
\ProcessOptions\relax

\DeclareOption{11pt}{\OptionNotUsed}
\DeclareOption{12pt}{\OptionNotUsed}
\DeclareOption{twocolumn}{\OptionNotUsed}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}

\ProcessOptions\relax

\LoadClass[10pt,a4paper,twoside]{article}

\RequirePackage{amsmath,amsfonts,amssymb,amsthm,amsbsy,amsmath}
\RequirePackage{bm}
\RequirePackage{engord}
\RequirePackage{epsfig,vmargin,booktabs}
\RequirePackage[T1]{fontenc}
\RequirePackage[a4paper]{geometry}  % required to get exact A4 size
\RequirePackage{graphicx}
\RequirePackage{lastpage}
\RequirePackage{latexsym}
\RequirePackage{natbib}
\RequirePackage{siunitx}
\RequirePackage{tabularx}
\RequirePackage{tikz}
\RequirePackage{times}
\RequirePackage{verbatim}

%\setmarginsrb{left}{top}{right}{bottom}{headh1gt}{headsep}{foothgt}{footskip}
\setmarginsrb{12.7mm}{25mm}{12.7mm}{19mm}{18mm}{4mm}{6mm}{10mm}
% headsep:  Distance from bottom of header to the body of text on a page.
% footskip: Distance from bottom of body to the bottom of the footer (the footer is the text at the bottom of each page).

\parindent=0pt
%\parskip0pc
\parskip=2ex

% variable definitions
\def\doi#1{\def\@doi{#1}}\newcommand{\printdoi}{\@doi}
\def\year#1{\def\@year{#1}}\newcommand{\printyear}{\@year}
\def\editor#1{\def\@editor{#1}}\newcommand{\printeditor}{\@editor}
\def\submitteddate#1{\def\@submitteddate{#1}}\newcommand{\printsubmitteddate}{\@submitteddate}
\def\accepteddate#1{\def\@accepteddate{#1}}\newcommand{\printaccepteddate}{\@accepteddate}
\def\publisheddate#1{\def\@publisheddate{#1}}\newcommand{\printpublisheddate}{\@publisheddate}
\def\citation#1{\def\@citation{#1}}\newcommand{\printcitation}{\@citation}
\def\authorfooter#1{\def\@authors{#1}}\newcommand{\printauthors}{\@authors}
\def\issn#1{\def\@issn{#1}}\newcommand{\printissn}{\@issn}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The LABEL extensions for proofing. J. Kaipio, 4.4.1995
%% CAPTIONS FOR TABLES AND FIGURES
%%
%%  There's a new logical variable \tablecap, which is
%%  set \true by saying \tablecaptrue and \false by
%%  \tablecapfalse. The variable is expanded by saying
%%  \iftablecap. Then there are two separate \caption
%%  macros: \tabcaption and \figcaption, the definition
%%  of which are obvious. The \iftablecap is performed in
%%  \@makecaption, which now contains two different
%%  captioning styles.
%%                               J. Kaipio 3.10.1990.
\newif\iftablecap
\def\tabcaption#1{\tablecaptrue\caption{#1}}
\def\figcaption#1{\tablecapfalse\caption{#1}}
\newlength{\t@bindwidth} \setlength{\t@bindwidth}{1cm}
\newlength{\c@pwidth} \setlength{\c@pwidth}{\hsize}
\addtolength{\c@pwidth}{-2\t@bindwidth}

\long\def\@makecaption#1#2{
   %\vskip 10pt
   \vskip 0.5ex
   \iftablecap                           % Is it a table?
     \setbox\@tempboxa\hbox{#2}
     \ifdim \wd\@tempboxa > \c@pwidth      % IF longer than one line-2cm:
         \hbox to\hsize{\hfil{\sc #1}\hfil}
         \begin{list}{}{\leftmargin\t@bindwidth
             \rightmargin\t@bindwidth}
             {\item \footnotesize  #2}
         \end{list}
     \else                               % ELSE  center.
         \hbox to\hsize{\hfil{\sc #1}\hfil}
         \hbox to\hsize{\hfil{\footnotesize #2}\hfil}
     \fi                                   % of \ifdim
   \else                                 % Not a table: a picture
     \setbox\@tempboxa\hbox{\small{\bf #1}. #2}
     \ifdim \wd\@tempboxa >\c@pwidth       % IF longer than one line-2cm:
        \begin{list}{}{\leftmargin\t@bindwidth
           \rightmargin\t@bindwidth}
           {\small \item {\bf #1}. #2}
         \end{list}                        % THEN set as ordinary paragraph.
     \else                               % ELSE  center.
         \hbox to\hsize{\hfil\box\@tempboxa\hfil}
     \fi                                   % of \ifdim
   \fi                                   % of \if\tablecaptrue
   }                                     % \@makecaption

%  End of \@makecaption modifications.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% MAKETITLE
%%
%% Overwriting \maketitle to prevent floats from slipping above a manually
%% created title
%%
%% CMS, 10.01.2023

%% Authorlist
%% Print comma-separated authorlist after calling
%% /addauthor{authorname}{affiliationID} multiple times
%% Adapted from: https://tex.stackexchange.com/a/231056 (11.01.2023)
\newcommand{\printlist}[2][,]{{% Print list
		% http://tex.stackexchange.com/a/89187/5764
		\def\listsep{\def\listsep{#1}}% Delayed execution of list separator
		\renewcommand{\do}[1]{\listsep~##1}%
		\dolistloop\authorlist}}
\newcommand{\addauthor}[2]{
	\listadd{\authorlist}{#1\textsuperscript{#2}}}

%% Affiliationlist
%% Print affiliations as a vertical list after calling
%% /addaffiliation{affiliationID}{affilationName}{EmailOrchidString}
%% multiple times
%% Adapted from: https://tex.stackexchange.com/a/47352 (11.01.2023)
\newcommand{\affiliationlist}{}
\newcommand{\addaffiliation}[3]{
	\makeatletter
	\expandafter\def\expandafter\affiliationlist\expandafter{\affiliationlist{}
		\textsuperscript{#1}~#2;~#3\\}
	\makeatother}


%% Overwrite maketitle
\renewcommand{\@maketitle}{%

  \noindent\makebox[\linewidth]{\rule{\textwidth}{1.0pt}}

  \begin{flushleft}
    {\fontsize{9pt}{11pt}\selectfont%
      % Do not edit.
      Type of the Paper: Conference Paper
    }
  \end{flushleft}

  \begin{flushleft}
    {\fontsize{18pt}{20pt}\selectfont%
      % Replace with your paper title.
      \@title
    }
  \end{flushleft}

  \begin{flushleft}
    {\fontsize{10}{12}\selectfont{
        % Replace with your authors.
        \printlist{\authorlist}
      }\\}
    \vspace{10pt}
    {\fontsize{8pt}{10pt}\selectfont{
        % Replace with your affiliations.
        %\textsuperscript{1} Affiliation 1; email One; ORCID One (if applicable) \\
        %\textsuperscript{2} Affiliation 2; email Two, ORCID Two (if applicable);
        %email Three, ORCID Three (if applicable) \\
        \affiliationlist
        \textsuperscript{*} corresponding author
    }}
  \end{flushleft}

  \noindent\makebox[\linewidth]{\rule{\textwidth}{0.4pt}}

  \begin{flushleft}
    {\fontsize{8pt}{10pt}\selectfont{
        Name of Editor: \printeditor \\
        Submitted: \printsubmitteddate \\
        Accepted: \printaccepteddate \\
        Published: \printpublisheddate \\
        Citation: \printcitation
    }}
  \end{flushleft}
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SECTIONS
\renewcommand\section{\@startsection {section}{1}{\z@}{-3.5ex plus -1ex minus-.2ex}{0.2ex}{\fontsize{11}{13}\bf}}
\renewcommand\subsection{\@startsection {subsection}{1}{\z@}{-3.5ex plus -1ex minus-.2ex}{0.2ex}{\fontsize{11}{13}\bf}}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%
% HEADERS & FOOTERS
%%%%%%%%%%%%%%%%%%%

% style for the first page
\renewcommand{\ps@plain}{%
  \renewcommand{\@oddhead}{{\parbox[b]{\textwidth}{
    % NOTE : if there are blank lines in the code here, it can cause line
    % breaks in the rendered document. I don't understand why that would occur.
    \begin{minipage}{0.5\textwidth}
      The Evolving Scholar Vol. 3, \printyear
    \end{minipage}
    \begin{minipage}{0.5\textwidth}
      \scriptsize \raggedleft http://dx.doi.org/\printdoi
    \end{minipage}
    \vspace{1mm}
    \\
    % logos are 1cm in height @ 300 ppi
    \begin{minipage}{0.2\textwidth}
      \includegraphics[height=1cm]{bmd2023-logo-256x118.png}
    \end{minipage}
    \begin{minipage}{0.2\textwidth}
      \includegraphics[height=1cm]{tu-delft-open-logo-294x118.png}
    \end{minipage}
    \begin{minipage}{0.6\textwidth}
      \scriptsize \raggedleft \textbf{Bicycle and Motorcycle Dynamics 2023\\
        Symposium on the Dynamics and Control of Single Track Vehicles\\
        18--20 October 2023, Delft University of Technology, The Netherlands}
    \end{minipage}
    \vspace{5mm}
  }}}
  \renewcommand{\@oddfoot}{{\parbox[b]{\textwidth}{
    \scriptsize%
    \copyright \printyear \hspace{1mm} \printauthors \hspace{1mm} published by TU Delft OPEN
    on behalf of the authors.\\%
    ISSN: \printissn
  }}}
}
% style for remaining pages after the first
\renewcommand{\ps@headings}{
  % reduce the headh1gt since the images are no longer present in the header
  \setmarginsrb{12.7mm}{25mm}{12.7mm}{19mm}{8mm}{4mm}{6mm}{10mm}
  %% headers
  \renewcommand{\@evenhead}{{\parbox[b]{\textwidth}{
    \begin{minipage}{0.5\textwidth}
      The Evolving Scholar Vol. 3, \printyear
    \end{minipage}
    \begin{minipage}{0.5\textwidth}
      \scriptsize \raggedleft http://dx.doi.org/\printdoi
    \end{minipage}
    \vspace{10mm}
  }}}
  \renewcommand{\@oddhead}{{\parbox[b]{\textwidth}{
    \begin{minipage}{0.5\textwidth}
      The Evolving Scholar Vol. 3, \printyear
    \end{minipage}
    \begin{minipage}{0.5\textwidth}
      \scriptsize \raggedleft http://dx.doi.org/\printdoi
    \end{minipage}
    \vspace{10mm}
  }}}
  %% footers
  \renewcommand{\@evenfoot}{{\parbox[b]{\textwidth}{
    \vspace{10mm}
    \begin{minipage}{0.75\textwidth}
      \scriptsize \raggedright
      \copyright \printyear \hspace{1mm} \printauthors \hspace{1mm} published by TU Delft OPEN
      on behalf of the authors.\\%
      ISSN: \printissn
    \end{minipage}
    \begin{minipage}{0.25\textwidth}
      \scriptsize \raggedleft \thepage\ of \pageref{LastPage}
    \end{minipage}
  }}}
  \renewcommand{\@oddfoot}{{\parbox[b]{\textwidth}{
    \vspace{10mm}
    \begin{minipage}{0.75\textwidth}
      \scriptsize \raggedright
      \copyright \printyear \hspace{1mm} \printauthors \hspace{1mm} published by TU Delft OPEN
      on behalf of the authors.\\%
      ISSN: \printissn
    \end{minipage}
    \begin{minipage}{0.25\textwidth}
      \scriptsize \raggedleft \thepage\ of \pageref{LastPage}
    \end{minipage}
  }}}
}

\AtBeginDocument{
  \pagestyle{plain}
}

\endinput
