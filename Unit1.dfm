object Form1: TForm1
  Left = 241
  Top = 155
  Width = 943
  Height = 546
  VertScrollBar.Range = 200
  ActiveControl = Button1
  Caption = 'Demo of Python'
  Color = clBtnFace
  CustomTitleBar.CaptionAlignment = taCenter
  Font.Charset = DEFAULT_CHARSET
  Font.Color = clWindowText
  Font.Height = 11
  Font.Name = 'MS Sans Serif'
  Font.Pitch = fpVariable
  Font.Style = []
  OldCreateOrder = True
  PixelsPerInch = 96
  TextHeight = 13
  object Splitter1: TSplitter
    Left = 0
    Top = 153
    Width = 927
    Height = 3
    Cursor = crVSplit
    Align = alTop
    Color = clBtnFace
    ParentColor = False
    ExplicitWidth = 536
  end
  object Memo1: TMemo
    Left = 0
    Top = 156
    Width = 927
    Height = 307
    Align = alClient
    Font.Charset = DEFAULT_CHARSET
    Font.Color = clWindowText
    Font.Height = -13
    Font.Name = 'Consolas'
    Font.Pitch = fpVariable
    Font.Style = []
    Lines.Strings = (
      'import sys'
      'if '#39'd:\\gitRepos\\shortTrackScheduler'#39' not in sys.path:'
      '    sys.path.append('#39'd:\\gitRepos\\shortTrackScheduler'#39')'
      'from schedule import raceProgram'
      ''
      'raceProgram_ = raceProgram(totalSkaters=18,'
      '                           numRacesPerSkater=3,'
      '                           heatSize=5,'
      '                           considerSeeding=False,'
      '                           fairStartLanes=True,'
      '                           minHeatSize=3,'
      '                           printDetails=True,'
      '                           cleanCalculationDetails=True'
      '                          )'
      'heatDict = raceProgram_.buildHeats(adjustAfterNAttempts=2000,'
      '                                   method='#39'sgp'#39')'
      '')
    ParentFont = False
    ScrollBars = ssBoth
    TabOrder = 1
  end
  object Panel1: TPanel
    Left = 0
    Top = 463
    Width = 927
    Height = 44
    Align = alBottom
    BevelOuter = bvNone
    TabOrder = 0
    object Button1: TButton
      Left = 6
      Top = 8
      Width = 115
      Height = 25
      Caption = 'Execute script'
      TabOrder = 0
      OnClick = Button1Click
    end
    object Button2: TButton
      Left = 168
      Top = 8
      Width = 91
      Height = 25
      Caption = 'Load script...'
      TabOrder = 1
      OnClick = Button2Click
    end
    object Button3: TButton
      Left = 264
      Top = 8
      Width = 89
      Height = 25
      Caption = 'Save script...'
      TabOrder = 2
      OnClick = Button3Click
    end
  end
  object Memo2: TMemo
    Left = 0
    Top = 0
    Width = 927
    Height = 153
    Align = alTop
    Font.Charset = DEFAULT_CHARSET
    Font.Color = clWindowText
    Font.Height = -13
    Font.Name = 'Consolas'
    Font.Pitch = fpVariable
    Font.Style = []
    ParentFont = False
    ScrollBars = ssBoth
    TabOrder = 2
    ExplicitLeft = 8
    ExplicitTop = -3
  end
  object PythonEngine1: TPythonEngine
    DllName = 'python39.dll'
    APIVersion = 1013
    RegVersion = '3.9'
    UseLastKnownVersion = False
    IO = PythonGUIInputOutput1
    Left = 32
  end
  object OpenDialog1: TOpenDialog
    DefaultExt = '*.py'
    Filter = 'Python files|*.py|Text files|*.txt|All files|*.*'
    Title = 'Open'
    Left = 176
  end
  object SaveDialog1: TSaveDialog
    DefaultExt = '*.py'
    Filter = 'Python files|*.py|Text files|*.txt|All files|*.*'
    Title = 'Save As'
    Left = 208
  end
  object PythonGUIInputOutput1: TPythonGUIInputOutput
    UnicodeIO = True
    RawOutput = False
    Output = Memo2
    Left = 112
    Top = 32
  end
end
